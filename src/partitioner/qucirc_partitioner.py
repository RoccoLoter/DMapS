import math
import kahypar
from copy import deepcopy
from bidict import bidict
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple
from qiskit.circuit.gate import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit


from global_config import repo_path


class QuCircPartitioner:
    """
    Quantum circuit partitioner based on classical graph partition algorithm.
    """

    def __init__(self) -> None:
        self.k = None

        self._chips_tmp_idx = bidict()
        self._consider_nodes_wgt = None
        self._ops_tag = {}

    def run(
        self,
        quantum_circuit: QuantumCircuit,
        chips_scale: Dict[int, int] = None,
        capacity_const: bool = True,
        consider_nodes_weight: bool = False,
        k: int = None,
    ) -> Dict:
        par_res = {}  # Partition result

        # Remove barrier and measure operation from the origin quantum circuit
        qu_dag = self._remove_barrier_measure(quantum_circuit)

        if capacity_const:
            self.k = len(chips_scale)
        else:
            self.k = k
        self._consider_nodes_wgt = consider_nodes_weight

        (
            qubits_tmp_idx,
            num_vers,
            num_hyEdges,
            hyperedges_idx,
            hyperedges,
            nodes_weight,
            edges_weight,
        ) = self._build_circ_hyperG(qu_dag)

        print("num_vers: ", num_vers)
        print("num_hyEdges: ", num_hyEdges)
        print("hyperedge_indices: ", hyperedges_idx)
        print("hyperedges: ", hyperedges)
        print("node_weights: ", nodes_weight)
        print("edge_weight: ", edges_weight)

        hyperG = kahypar.Hypergraph(
            num_vers,
            num_hyEdges,
            hyperedges_idx,
            hyperedges,
            self.k,
            edges_weight,
            nodes_weight,
        )

        # Load the config file
        config_fp_str = str(
            repo_path / "src" / "partitioner" / "config" / "km1_kKaHyPar_sea20.ini"
        )
        hyperG_context = kahypar.Context()
        hyperG_context.loadINIconfiguration(config_fp_str)
        hyperG_context.setK(self.k)
        hyperG_context.suppressOutput(True)

        # Set the custom target block weights of hypergraph partition.
        init_partition = [0] * num_vers
        if capacity_const:
            # Calculate the maximum node weight capacity of each chip.
            chips_capacity = self._calculate_chips_capacity(
                quantum_circuit, chips_scale
            )

            #! The setting of parameters needs to be further improved. Different quantum programs require different parameter designs.
            idx = 0  # The temporary index of each chip
            blocks_cap = []
            for chip_idx, chip_cap in chips_capacity.items():
                blocks_cap.append(chip_cap)
                self._chips_tmp_idx[chip_idx] = idx
                idx += 1

            hyperG_context.setCustomTargetBlockWeights(blocks_cap)

            # TODO: Need to be optimized
            # Set the initial partition of hypergraph
            sorted_qubit_tmp_idx = {qubit : qubits_tmp_idx[qubit] for qubit in quantum_circuit.qubits}
            sorted_tmp_idx = list(sorted_qubit_tmp_idx.values())
            
            tmp_dict = {}
            block_idx = 0
            blocks_cap_bak = deepcopy(blocks_cap)
            for tmp_idx in sorted_tmp_idx:
                if blocks_cap_bak[block_idx] == 0:
                    block_idx += 1

                tmp_dict[tmp_idx] = block_idx
                blocks_cap_bak[block_idx] -= 1
            
            for k, v in tmp_dict.items():
                init_partition[k] = v
            
            # hyperG_context.setInitialPartition(init_partition)
                

        hyperG_context.setEpsilon(0.01)
        kahypar.partition(hyperG, hyperG_context) 
        hyperG.printGraphState()

        # Update the partition result
        for qubit, tmp_id in qubits_tmp_idx.items():
            par_res[qubit] = self._chips_tmp_idx.inverse[hyperG.blockID(tmp_id)]

        return par_res

    def _build_circ_hyperG(
        self, dag: DAGCircuit
    ) -> Tuple[bidict, int, int, List[int], List[int], List[int], List[int]]:
        """
        Build the hypergraph data of given quantum circuit.
        """
        num_vertices = 0  # The number of vertices
        num_hyperedges = 0  # The number of hyperedges
        hyperedges = []  # The list of hyperedges
        vertices_weight = []  # The list of nodes weight
        edges_weight = []  # The list of hyperedges weight
        hyperedges_idx = []  # The list of hyperedges index
        qubits_tmp_idx = bidict()  # The dict of index and node that without zero weight

        # The temporary variables
        vers_wgt_dict = {}  # The dict of used qubits and its weight
        ori_edges_wgt_dict = {}  # The dict of hyperedges and its weight

        edges_wgt_dict = {}
        twoQ_ops_blocks = self._find_2q_op_block(dag)
        # Calculate the weight of node and net
        for ops_block in twoQ_ops_blocks:
            qubits = []
            for op in ops_block:
                ctl_Q, tgt_Q = op.qargs[0], op.qargs[1]

                # Update the weight of used virtual qubits
                if self._consider_nodes_wgt:
                    # Update the weight of control qubit
                    if ctl_Q not in vers_wgt_dict:
                        vers_wgt_dict[ctl_Q] = 1
                    else:
                        vers_wgt_dict[ctl_Q] += 1

                    # Update the weight of target qubit
                    if tgt_Q not in vers_wgt_dict:
                        vers_wgt_dict[tgt_Q] = 1
                    else:
                        vers_wgt_dict[tgt_Q] += 1
                else:
                    # Update the weight(equal to 1) of control qubit
                    if ctl_Q not in vers_wgt_dict:
                        vers_wgt_dict[ctl_Q] = 1

                    # Update the weight(equal to 1) of target qubit
                    if tgt_Q not in vers_wgt_dict:
                        vers_wgt_dict[tgt_Q] = 1

                # Add the control qubit and target qubit to the hyperedge
                if ctl_Q not in qubits:
                    qubits.append(ctl_Q)
                if tgt_Q not in qubits:
                    qubits.append(tgt_Q)

            if tuple(qubits) not in ori_edges_wgt_dict:
                ori_edges_wgt_dict[tuple(qubits)] = len(ops_block)
            else:
                ori_edges_wgt_dict[tuple(qubits)] += len(ops_block)

        # Create the temporary index of each qubit
        # Add the weight of each qubit to the list of nodes weight
        tmp_idx = 0
        for qubit, weight in vers_wgt_dict.items():
            qubits_tmp_idx[qubit] = tmp_idx
            tmp_idx += 1
            vertices_weight.append(weight)

        # Update the weight of net
        for qubits, weight in ori_edges_wgt_dict.items():
            index_tuple = []  # The list of qubit's temporary index

            for qubit in qubits:
                index_tuple.append(qubits_tmp_idx[qubit])
            index_tuple.sort()

            if tuple(index_tuple) not in edges_wgt_dict:
                edges_wgt_dict[tuple(index_tuple)] = weight
            else:
                edges_wgt_dict[tuple(index_tuple)] += weight

        # Build the information of hyperedges, including:
        # 1. The index of hyperedge
        # 2. The weight of hyperedge
        hyperE_index = 0
        for index_tuple, weight in edges_wgt_dict.items():
            for index in index_tuple:
                hyperedges.append(index)
            edges_weight.append(weight)

            hyperedges_idx.append(hyperE_index)
            hyperE_index += len(index_tuple)
        hyperedges_idx.append(hyperE_index)

        num_hyperedges = len(edges_wgt_dict)
        num_vertices = len(vers_wgt_dict)

        return (
            qubits_tmp_idx,
            num_vertices,
            num_hyperedges,
            hyperedges_idx,
            hyperedges,
            vertices_weight,
            edges_weight,
        )

    def _find_2q_op_block(self, dag: DAGCircuit) -> List[List[DAGOpNode]]:
        """
        Find the blocks information of two-qubit operations.
        """
        # The element of ``twoQ_ops_blocks`` is a list, which stores the information of two-qubit operations block that the quantum operation node belongs to.
        twoQ_ops_blocks = []

        # The information of whether the two-qubit quantum operation node is in the hypergraph
        self._ops_tag = {op: 0 for op in dag.two_qubit_ops()}

        # # Get all the operation nodes and their index that are acts on the qubits in the quantum dag
        op_nodes_on_qubits = self._gates_on_qubits(dag)

        for qubit, op_nodes_on_qubit in op_nodes_on_qubits.items():
            block_head = 0
            block_tail = block_head + 1

            num_ops = len(op_nodes_on_qubit)
            # The last quantum operation that acts on the specific qubit should be ignored.
            while block_head < num_ops - 1:
                head_op = op_nodes_on_qubit[block_head]

                if len(head_op.qargs) == 2:
                    # If the quantum operation node has not been constructed into the hypergraph, then it should judge the node whether could be added to the hypergraph.
                    if self._ops_tag[head_op] == 0:
                        twoQ_op_block = []
                        twoQ_op_block.append(head_op)

                        # Iterate through all quantum operation nodes after the head operation node, and judge whether the tail operation node whether can be added to the two-qubit operations block
                        while block_tail < num_ops:
                            tail_op = op_nodes_on_qubit[block_tail]
                            front_tail_op = op_nodes_on_qubit[block_tail - 1]

                            is_adjacent = self._judge_adjacent_op(
                                qubit, front_tail_op, tail_op, dag
                            )
                            # If the tail quantum operation node is adjacent to the head quantum operation node, then it should be added to the two-qubit operations block.
                            # And update the value of ``tail``.
                            if is_adjacent:
                                twoQ_op_block.append(tail_op)

                                # If block_tail is equal to ``num_ops - 1``, then the tail quantum operation node is the last quantum operation node, and it should be added to the two-qubit operations block.
                                if block_tail == num_ops - 1:
                                    if self._judge_valid_block(twoQ_op_block):
                                        new_block = [
                                            op
                                            for op in twoQ_op_block
                                            if len(op.qargs) == 2
                                        ]
                                        twoQ_ops_blocks.append(new_block)

                                        for op_node in new_block:
                                            self._ops_tag[op_node] = 1

                                    block_head = num_ops - 1
                                    break

                                block_tail += 1
                            else:
                                # If the number of quantum operation nodes in the two-qubit operations block is greater than 1, then it should be added to the list of two-qubit operations block. And update the state of quantum operation node in the ``twoQ_op_block``.
                                if self._judge_valid_block(twoQ_op_block):
                                    new_block = [
                                        op for op in twoQ_op_block if len(op.qargs) == 2
                                    ]
                                    twoQ_ops_blocks.append(new_block)

                                    for op_node in new_block:
                                        self._ops_tag[op_node] = 1

                                # If the tail quantum operation node is not adjacent to the head quantum operation node, then it should be assign the value of tail to head.
                                block_head = block_tail
                                block_tail = block_head + 1
                                break
                    else:
                        block_head += 1
                        block_tail = block_head + 1

                # If the quantum operation node is a single-qubit operation, then it should be ignored, and value of ``head`` and ``tail`` should be updated.
                elif len(head_op.qargs) == 1:
                    block_head += 1
                    block_tail = block_head + 1
                else:
                    raise ValueError(
                        "The number of qubits that quantum operation acts on are more than 2, this module can not work!"
                    )

        # TODO: The following code maybe need to be optimized.
        for op_node, op_tag in self._ops_tag.items():
            if op_tag == 0:
                ops_block = [op_node]
                twoQ_ops_blocks.append(ops_block)

        return twoQ_ops_blocks

    def _gates_on_qubits(self, dag: DAGCircuit) -> Dict[Qubit, List[DAGOpNode]]:
        """Find the gates that act on each qubit in the circuit."""
        gates_on_qubits = {}

        for qubit in dag.qubits:
            op_nodes = dag.nodes_on_wire(qubit, only_ops=True)

            # Filter out the barriers and measures
            gates_on_qubit = [node for node in op_nodes if isinstance(node.op, Gate)]

            if len(gates_on_qubit) > 0:
                gates_on_qubits[qubit] = gates_on_qubit
        return gates_on_qubits

    def _judge_adjacent_op(
        self,
        qubit: Qubit,
        front_op: DAGOpNode,
        current_op: DAGOpNode,
        dag: DAGCircuit,
    ):
        """Judge whether the operation node is adjacent to front operation node"""
        is_adjacent = False

        if len(current_op.qargs) == 1:
            return True

        pre_ops = []
        pre_ops_name = []
        for pre_op in dag.predecessors(current_op):
            if isinstance(pre_op, DAGOpNode):
                pre_ops.append(pre_op)
                pre_ops_name.append(pre_op.name)

        if len(pre_ops) == 1:
            # If the current operation node has only one predecessor(The front operation node), then it should be adjacent to the front operation node.
            is_adjacent = True
        elif len(pre_ops) == 2:
            # If the current operation node has two predecessors.
            # Then it should be judge the predecessor whether is adjacent to the front operation node.
            fst_pre_op, snd_pre_op = pre_ops[0], pre_ops[1]

            if fst_pre_op._node_id != front_op._node_id:
                if len(fst_pre_op.qargs) == 1:
                    # If the first predecessor is a single-qubit operation node,  then the search forward needs to continue...
                    op_list = [
                        op
                        for op in dag.predecessors(fst_pre_op)
                        if isinstance(op, DAGOpNode)
                    ]

                    if len(op_list) == 1:
                        tmp_op = op_list[0]

                        while tmp_op:
                            if len(tmp_op.qargs) == 1:
                                op_list = [
                                    op
                                    for op in dag.predecessors(tmp_op)
                                    if isinstance(op, DAGOpNode)
                                ]

                                if len(op_list) == 1:
                                    tmp_op = op_list[0]
                                    continue
                                else:
                                    is_adjacent = True
                                    break
                            elif len(tmp_op.qargs) == 2:
                                qubits_idx = [qubit.index for qubit in tmp_op.qargs]
                                if qubit.index in qubits_idx:
                                    # This predecessor quantum opertion is adjacent to the front operation node, because the qubits that operation acts on are the same as the qubit that current operation acts on.
                                    is_adjacent = True
                                else:
                                    is_adjacent = False
                                break

                    elif len(op_list) == 0:
                        is_adjacent = True
                    else:
                        raise ValueError("The number of predecessors is more than 2!")

                elif len(fst_pre_op.qargs) == 2:
                    # If the first predecessor is a two-qubit operation node, then it could not be adjacent to the front operation node.
                    is_adjacent = False

            if snd_pre_op._node_id != front_op._node_id:
                if len(snd_pre_op.qargs) == 1:
                    # If the second predecessor is a single-qubit operation node,  then the search forward needs to continue...
                    op_list = [
                        op
                        for op in dag.predecessors(snd_pre_op)
                        if isinstance(op, DAGOpNode)
                    ]

                    if len(op_list) == 1:
                        tmp_op = op_list[0]

                        while tmp_op:
                            if len(tmp_op.qargs) == 1:
                                op_list = [
                                    op
                                    for op in dag.predecessors(tmp_op)
                                    if isinstance(op, DAGOpNode)
                                ]

                                if len(op_list) == 1:
                                    tmp_op = op_list[0]
                                    continue
                                else:
                                    is_adjacent = True
                                    break
                            elif len(tmp_op.qargs) == 2:
                                qubits_idx = [qubit.index for qubit in tmp_op.qargs]
                                if qubit.index in qubits_idx:
                                    # This predecessor quantum opertion is adjacent to the front operation node, because the qubits that operation acts on are the same as the qubit that current operation acts on.
                                    is_adjacent = True
                                else:
                                    is_adjacent = False
                                break
                    elif len(op_list) == 0:
                        is_adjacent = True
                    else:
                        raise ValueError("The number of predecessors is more than 2!")

                elif len(snd_pre_op.qargs) == 2:
                    # If the second predecessor is a two-qubit operation node, then it could not be adjacent to the front operation node.
                    is_adjacent = False
        else:
            raise ValueError(
                "The number of predecessors of two-qubit operation can not be 2!"
            )

        return is_adjacent

    def _judge_valid_block(self, ops_block: List[DAGOpNode]) -> bool:
        """Determine the quantum operation block is valided."""
        is_added = False

        num_twoQ_op = 0
        for op in ops_block:
            if len(op.qargs) == 2:
                num_twoQ_op += 1

        if num_twoQ_op > 1:
            is_added = True

        return is_added

    def _remove_barrier_measure(self, quantum_circuit: QuantumCircuit) -> DAGCircuit:
        """Remove barrier and measure operation node from the origin quantum circuit."""
        origin_dag = circuit_to_dag(quantum_circuit)

        # Remove barrier and measure operation
        op_nodes = origin_dag.op_nodes()
        for op_node in op_nodes:
            if op_node.name == "barrier" and op_node.name == "measure":
                origin_dag.remove_op_node(op_node)

        new_circ = dag_to_circuit(origin_dag)
        new_dag = circuit_to_dag(new_circ)

        return new_dag

    def _calculate_chips_capacity(
        self, quantum_circuit: QuantumCircuit, chips_scale: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Calculate the maximum node weight capacity of each chip.

        Args:
            quantum_circuit(QuantumCircuit): Quantum circuit
            chip_scale_info(dict): The number of physical qubits of each quantum chip

        Return:
            chip_node_weight_capacity(dict): The maximum node weight capacity of each chip
        """
        chips_cap = {}

        if self._consider_nodes_wgt:
            nodes_weight = {qubit: 0 for qubit in quantum_circuit.qubits}
            weight_list = nodes_weight.values()

            for insn in quantum_circuit:
                if insn[0].name in self._2q_ops_name:
                    nodes_weight[insn[1][0]] += 1
                    nodes_weight[insn[1][1]] += 1

            # Calculate the average weight of node.
            num_used_qubits = 0
            total_weight = 0
            min_weight = math.inf

            for weight in weight_list:
                if weight > 0:
                    if weight < min_weight:
                        min_weight = weight
                    num_used_qubits += 1
                    total_weight += weight
            ave_weight = total_weight / num_used_qubits

            # Update the node weight capacity of each quantum chip.
            for idx, cap in chips_scale.items():
                #! For quantum programs with a particular structure, perhaps there are some problems here.
                # Round the value up
                chips_cap[idx] = math.ceil(cap * ave_weight)
        else:
            chips_cap = chips_scale

        return chips_cap
