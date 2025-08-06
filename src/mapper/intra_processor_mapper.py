from bidict import bidict
from copy import deepcopy
from multiprocessing import Pool
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.converters import circuit_to_dag
from qiskit.circuit import Qubit, QuantumRegister

from mapper.local_sabre_mapper import MultiModeSwap


class IntraProcessMapper:
    """
    The local qubit mapping within each quantum processor.
    """

    def __init__(self):
        self.circuit = None
        self.par_res = {}
        self.alloc_info = {}
        self.dist_matrix_tolist = None
        self.total_coupling_map = None
        self.ep_coupling_map = None
        self.ep_phy_qubits = {}
        self.chips_nearest_commu_qubits = {}
        self.ep_phy_qubit_n_idx = None
        self.ep_local_qubits = None
        self.ep_commu_qubits = None
        self.ep_local_qubit_n_idx = None
        self.ep_commu_qubit_n_idx = None
        self.intra_chip_all2all = False

    def run(
        self,
        total_circuit: QuantumCircuit,
        par_res: Dict[Qubit, int],
        alloc_info: bidict,
        dist_matrix_tolist: List,
        total_coupling_map: CouplingMap,
        ep_coupling_map: Dict,
        ep_phy_qubits: Dict[int, List[int]],
        chips_nearest_commu_qubits: Dict[Tuple[int], Tuple[int]],
        iter_num: int = 5,
        lookahead_ability: int = 20,
        intra_chip_all2all: bool = False,
    ) -> Dict[Qubit, int]:
        """
        Map the virtual qubits of  subcircuit to the physical qubits in the processor.
        """
        total_mapping_res = {}

        # Update the global variables.
        self.circuit = total_circuit
        self.par_res = par_res
        self.alloc_info = alloc_info
        self.dist_matrix_tolist = dist_matrix_tolist
        self.total_coupling_map = total_coupling_map
        self.ep_coupling_map = ep_coupling_map
        self.ep_phy_qubits = ep_phy_qubits
        self.chips_nearest_commu_qubits = chips_nearest_commu_qubits
        self.intra_chip_all2all = intra_chip_all2all

        (
            self.ep_phy_qubit_n_idx,
            self.ep_commu_qubit_n_idx,
            self.ep_local_qubit_n_idx,
            self.ep_commu_qubits,
            self.ep_local_qubits,
            subcircs,
        ) = self._local_mapping_info()

        if not self.intra_chip_all2all:
            args_pool = []
            for block_id, subcirc in subcircs.items():
                phy_qubit_n_idx = self.ep_phy_qubit_n_idx[self.alloc_info[block_id]]
                local_init_layout, commu_qubit_layout = self._initial_layout(block_id)
                local_coupling_map = self.ep_coupling_map[self.alloc_info[block_id]]

                args = [
                    block_id,
                    deepcopy(subcirc),
                    deepcopy(commu_qubit_layout),
                    deepcopy(local_init_layout),
                    self.dist_matrix_tolist,
                    self.total_coupling_map,
                    local_coupling_map,
                    deepcopy(phy_qubit_n_idx),
                    iter_num,
                    lookahead_ability,
                ]
                args_pool.append(args)
            pool = Pool()
            pool_res = pool.map(self._local_mapping_job, args_pool)

            # For each qubit block, update the total mapping result.
            for block_id, res in pool_res:
                local_qubit_n_idx = self.ep_local_qubit_n_idx[block_id]
                local_idx_list = [idx for _, idx in local_qubit_n_idx.items()]
                phy_qubit_n_idx = self.ep_phy_qubit_n_idx[self.alloc_info[block_id]]

                for tmp_vir_qubit, phy_id in res.items():
                    tmp_idx = tmp_vir_qubit.index
                    if tmp_idx in local_idx_list:
                        vir_qubit = local_qubit_n_idx.inverse[tmp_idx]
                        total_mapping_res[vir_qubit] = phy_qubit_n_idx.inverse[phy_id]
        else:
            # The intra-chip all-to-all connectivity mapping.
            for block_id, subcirc in subcircs.items():
                local_init_layout, _ = self._initial_layout(block_id)

                local_qubit_n_idx = self.ep_local_qubit_n_idx[block_id]
                local_idx_list = [idx for _, idx in local_qubit_n_idx.items()]
                phy_qubit_n_idx = self.ep_phy_qubit_n_idx[self.alloc_info[block_id]]
                for tmp_vir_qubit, phy_id in local_init_layout._v2p.items():
                    tmp_idx = tmp_vir_qubit.index
                    if tmp_idx in local_idx_list:
                        vir_qubit = local_qubit_n_idx.inverse[tmp_idx]
                        total_mapping_res[vir_qubit] = phy_qubit_n_idx.inverse[phy_id]  

        return total_mapping_res

    def _initial_layout(
        self,
        block_id: int,
    ):
        """
        Initialize the layout of the subcircuit.
        """
        init_layout = Layout()
        commu_qubit_layout = {}

        chip_id = self.alloc_info[block_id]
        phy_qubits = self.ep_phy_qubits[chip_id]
        phy_qubit_n_idx = self.ep_phy_qubit_n_idx[chip_id]
        local_qubits = self.ep_local_qubits[block_id]
        local_qubits_n_idx = self.ep_local_qubit_n_idx[block_id]
        commu_qubits = self.ep_commu_qubits[block_id]
        commu_qubits_n_idx = self.ep_commu_qubit_n_idx[block_id]

        # The list of total physical data qubits that can be used in the quantum processor.
        tmp_phy_qubits = deepcopy(list(phy_qubit_n_idx.values()))
        idx_list = [i for i in range(len(phy_qubits))]

        local_reg = QuantumRegister(name="lq", size=len(phy_qubits))
        commu_reg = QuantumRegister(name="cq", size=len(commu_qubits))

        for c_q in commu_qubits:
            other_chip_id = self.alloc_info[self.par_res[c_q]]
            chip_pair = (chip_id, other_chip_id)
            rev_chip_pair = (other_chip_id, chip_id)

            phy_commu_qubits = []
            if chip_pair in self.chips_nearest_commu_qubits:
                nearest_commu_qubits = self.chips_nearest_commu_qubits[chip_pair]
                phy_commu_qubits = [
                    qubit_pair[0] for qubit_pair in nearest_commu_qubits
                ]
            elif rev_chip_pair in self.chips_nearest_commu_qubits:
                nearest_commu_qubits = self.chips_nearest_commu_qubits[rev_chip_pair]
                phy_commu_qubits = [
                    qubit_pair[1] for qubit_pair in nearest_commu_qubits
                ]

            new_qubit = Qubit(register=commu_reg, index=commu_qubits_n_idx[c_q])
            commu_qubit_layout[new_qubit] = phy_qubit_n_idx[phy_commu_qubits[0]]

        # The virtual qubits that mapped to the quantum processor.
        qubit_map_res = {}
        for l_q in local_qubits:
            idx = local_qubits_n_idx[l_q]
            new_qubit = Qubit(register=local_reg, index=idx)
            qubit_map_res[new_qubit] = tmp_phy_qubits.pop(0)
            idx_list.remove(idx)
        if len(idx_list) > 0:
            for idx in idx_list:
                new_qubit = Qubit(register=local_reg, index=idx)
                qubit_map_res[new_qubit] = tmp_phy_qubits.pop(0)
        init_layout.from_dict(input_dict=qubit_map_res)

        return init_layout, commu_qubit_layout

    def _local_mapping_job(self, args):
        block_id = args[0]
        subcirc = args[1]
        commu_qubit_layout = args[2]
        input_layout = args[3]
        input_matrix_tolist = args[4]
        total_coupling_map = args[5]
        local_coupling_map = args[6]
        phy_qubit_n_idx = args[7]
        iter_num = args[8]
        lookahead_ability = args[9]

        local_mapping_res = self._sabre_mapping(
            circuit=subcirc,
            commu_qubit_layout=commu_qubit_layout,
            input_layout=input_layout,
            dist_matrix_tolist=input_matrix_tolist,
            total_coupling_map=total_coupling_map,
            local_coupling_map=local_coupling_map,
            phy_qubit_n_idx=phy_qubit_n_idx,
            iter_num=iter_num,
            lookahead_ability=lookahead_ability,
        )

        return block_id, local_mapping_res

    def _local_mapping_info(self):
        """
        Obtain the information of local mapping.
        """
        subcirc_info = {}

        ep_local_qubit = {}
        # Generate the information of qubit blocks.
        for qubit, block_id in self.par_res.items():
            if block_id in ep_local_qubit:
                ep_local_qubit[block_id].append(qubit)
            else:
                ep_local_qubit[block_id] = [qubit]
        op_blocks = {id: [] for id in ep_local_qubit.keys()}
        ep_commu_qubits = deepcopy(op_blocks)

        ep_phy_qubit_n_idx = {}
        for k, v in self.ep_phy_qubits.items():
            phy_qubit_n_idx = bidict({qubit: idx for idx, qubit in enumerate(v)})
            ep_phy_qubit_n_idx[k] = deepcopy(phy_qubit_n_idx)

        # Update the information of quantum operation blocks.
        for insn in self.circuit:
            if insn[0].name == "cx" or insn[0].name == "cz":
                qubits = insn.qubits
                ctl_qubit, tgt_qubit = qubits[0], qubits[1]
                ctl_block_id, tgt_block_id = (
                    self.par_res[ctl_qubit],
                    self.par_res[tgt_qubit],
                )

                if ctl_block_id != tgt_block_id:
                    ep_commu_qubits[ctl_block_id].append(tgt_qubit)
                    ep_commu_qubits[tgt_block_id].append(ctl_qubit)

                    # Add the quantum operation to the corresponding block.
                    op_blocks[ctl_block_id].append(insn)
                    op_blocks[tgt_block_id].append(insn)
                else:
                    block_id = ctl_block_id

                    # Add the quantum operation to the corresponding block.
                    op_blocks[block_id].append(insn)

        # For each local qubit, generate the corresponding temporary index.
        local_qubit_n_idx = {}
        for block_id, local_qubits in ep_local_qubit.items():
            qubit_n_idx = bidict()
            for index, qubit in enumerate(local_qubits):
                qubit_n_idx[qubit] = index
            local_qubit_n_idx[block_id] = deepcopy(qubit_n_idx)

        # For each remote qubit, generate the corresponding temporary index.
        commu_qubit_n_idx = {}
        for block_id, commu_qubits in ep_commu_qubits.items():
            qubit_n_idx = bidict()
            for index, qubit in enumerate(commu_qubits):
                qubit_n_idx[qubit] = index
            commu_qubit_n_idx[block_id] = deepcopy(qubit_n_idx)

        for block_id, op_block in op_blocks.items():
            local_qubits = ep_local_qubit[block_id]
            commu_qubits = ep_commu_qubits[block_id]
            num_phy_qubits = len(self.ep_phy_qubits[self.alloc_info[block_id]])
            num_commu_qubits = len(commu_qubits)

            new_circ = QuantumCircuit(
                name="block_" + str(block_id), global_phase=self.circuit.global_phase
            )
            local_register = QuantumRegister(name="lq", size=num_phy_qubits)
            commu_register = QuantumRegister(name="cq", size=num_commu_qubits)
            new_circ.add_register(local_register)
            new_circ.add_register(commu_register)

            for insn in op_block:
                instruction = insn[0]
                fst_qubit, snd_qubit = insn[1][0], insn[1][1]
                qubits = []
                if fst_qubit in commu_qubits and snd_qubit not in commu_qubits:
                    commu_qubit = Qubit(
                        register=commu_register,
                        index=commu_qubit_n_idx[block_id][fst_qubit],
                    )
                    local_qubit = Qubit(
                        register=local_register,
                        index=local_qubit_n_idx[block_id][snd_qubit],
                    )
                    qubits = [commu_qubit, local_qubit]
                elif fst_qubit not in commu_qubits and snd_qubit in commu_qubits:
                    local_qubit = Qubit(
                        register=local_register,
                        index=local_qubit_n_idx[block_id][fst_qubit],
                    )
                    commu_qubit = Qubit(
                        register=commu_register,
                        index=commu_qubit_n_idx[block_id][snd_qubit],
                    )
                    qubits = [local_qubit, commu_qubit]
                else:
                    qubits = [
                        Qubit(
                            register=local_register,
                            index=local_qubit_n_idx[block_id][qubit],
                        )
                        for qubit in insn[1]
                    ]

                clbits = [clbit for clbit in insn[2]]
                new_circ.append(instruction, qubits, clbits)

            subcirc_info[block_id] = deepcopy(new_circ)

        return (
            ep_phy_qubit_n_idx,
            commu_qubit_n_idx,
            local_qubit_n_idx,
            ep_commu_qubits,
            ep_local_qubit,
            subcirc_info,
        )

    def _sabre_mapping(
        self,
        circuit: QuantumCircuit,
        commu_qubit_layout: Dict,
        input_layout: Layout,
        dist_matrix_tolist: List,
        total_coupling_map: CouplingMap,
        local_coupling_map: CouplingMap,
        phy_qubit_n_idx: List[int],
        iter_num: int = 5,
        lookahead_ability: int = 20,
    ):
        """The mapping strategy of SABRE."""
        init_layout = input_layout
        circ_dag = circuit_to_dag(circuit)
        rev_circ = circuit.reverse_ops()
        dag = circ_dag
        rev_dag = circuit_to_dag(rev_circ)

        for _ in range(iter_num):
            for _ in ("forward", "backword"):
                router = MultiModeSwap(
                    total_coupling_map=total_coupling_map,
                    local_coupling_map=local_coupling_map,
                    fake_run=True,
                    commu_qubit_layout=commu_qubit_layout,
                    input_layout=init_layout,
                    input_matrix_tolist=dist_matrix_tolist,
                    phy_qubit_n_idx=phy_qubit_n_idx,
                    lookahead_ability=lookahead_ability,
                )
                final_layout = router.run(dag)

                # Update initial layout and reverse the circuit.
                init_layout = final_layout
                dag, rev_dag = rev_dag, dag

        final_mapping_res = deepcopy(init_layout._v2p)

        return final_mapping_res
