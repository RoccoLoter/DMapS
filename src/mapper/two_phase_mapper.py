import time
import random
import itertools
import numpy as np
import networkx as nx
from pathlib import Path
from copy import deepcopy
from bidict import bidict
from copy import deepcopy
from multiprocessing import Pool
from itertools import combinations
from typing import List, Any, Dict, Tuple

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.circuit import Qubit, CircuitInstruction

from partitioner.qucirc_partitioner import QuCircPartitioner
from frontend.chips_info_reader import QuHardwareInfoReader
from frontend.create_matrix import create_chip_dist_matrix
from mapper.intra_processor_mapper import IntraProcessMapper

FST_PHASE_DISTURBED_VALUE = 1
SEC_PHASE_DISTURBED_VALUE = 1

FST_LST_LEN = 20
SEC_LST_LEN = 20

EPR_COST = 5


class TwoPhaseMapper:
    """
    Get the initial mapping result of the distributed quantum circuit based on the hyper-heuristic algorithm.
    """

    def __init__(self, fst_phase_iter: int = 100, sec_phase_iter: int = 200) -> None:
        """
        Args:
            fst_phase_iter: The number of iterations of the first phase.
            sec_phase_iter: The number of iterations of the second phase.
        """
        # The number of iterations of the first phase(qubit group assignment) and the second phase(local quantum chip qubits initial mapping).
        self.fst_phase_iter = fst_phase_iter
        self.sec_phase_iter = sec_phase_iter

        # The quantum circuit partition result
        self.par_res = None
        self.best_par_res = None
        self.best_dis_num = None

        # The information of quantum circuit
        self.total_qc = None
        self.used_vir_qubits = None
        # The information about each quantum circuit partition.
        self.qubit_blocks = None
        self.commu_qubits = None
        self.op_blocks = None
        self.remote_ops = None

        # The information of quantum chips
        self.chips = None
        self.total_chip_cap_info = None  # The capacity information of all quantum chips in the quantum chip network.
        self.each_chip_commu_qubits = None

        self.run_time = None

        """
        Related information of the candidate quantum chip cluster.
        self.candidate_chip_cap_info: The capacity information of candidate quantum chips in the quantum chip network.
        self.chip_n_tmp_idx: The temporary index information of quantum chips.
        """
        self.chip_n_tmp_idx = None
        self.candidate_chip_cap_info = None
        self.intra_chip_all2all = False

        # The various matrices
        self.dist_matrix = None
        self.chip_dist_matrix = None
        # self.chips_commu_dist_matrix = None
        self.chips_nearest_commu_qubit = None
        self.nearest_phy_commu_qubits = None
        self.ep_physical_qubits = {}
        self.total_coupling_map = None
        self.ep_coupling_map = None

        self.best_assign = None
        self.best_mapping_res = None

    def run(
        self,
        quantum_circuit: QuantumCircuit,
        config_fp: Path,
        chip_type: str,
        dist_matrix: np.ndarray,
        local_mapping_mode: str = "hsa",
        is_random_assign=False,
        is_global_calculate=False,
        intra_chip_all2all: bool = False,
    ) -> bidict[Qubit, int]:
        self.dist_matrix = dist_matrix
        self.total_qc = quantum_circuit
        self.intra_chip_all2all = intra_chip_all2all

        chips_net_obj = QuHardwareInfoReader(config_fp)
        is_has_multi_chips, hardware_info = chips_net_obj.get_hardware_info(
            chip_type=chip_type
        )
        num_phy_qubits = len(hardware_info.get_total_qubits())
        self.used_vir_qubits = self._count_used_qubits(self.total_qc)
        num_vir_qubits = len(self.used_vir_qubits)

        # Make a simple judgment and check if there are more physical qubits than virtual qubits.
        if num_phy_qubits < num_vir_qubits:
            raise ValueError(
                "ERROR: The number of physical qubits that quantum chip network contains is smaller than the number of virtual qubits!"
            )

        start = time.time()
        # Judge whether there are multiple quantum chips in the quantum chip network.
        if is_has_multi_chips:
            # The information about quantum chip network.
            self.chips = hardware_info.chips
            chip_connections = hardware_info.chip_connections

            # Used for the local qubit mapping based on the SABRE algorithm.
            self.ep_physical_qubits, self.total_coupling_map, self.ep_coupling_map = (
                self._obtain_coupling_map(self.chips, chip_connections)
            )

            self.total_chip_cap_info = hardware_info.get_each_chip_capacity()
            self.each_chip_commu_qubits = hardware_info.get_each_chip_commu_qubits_idx()

            # Create the distance matrix of quantum chip to quantum chip
            chips_topology = hardware_info.obtain_chip_network()
            self.chip_dist_matrix = create_chip_dist_matrix(chips_topology)

            # Generate the information about quantum chip network
            self.chips_nearest_commu_qubit = self._create_commu_rout_info()
            self.nearest_phy_commu_qubits = self._obtain_nearest_commu_qubits()

            # Analyze all the chip cluster candidates.
            chips_candidate_info = self._analyze_chip_candidates(
                num_virtual_qubits=num_vir_qubits,
                chip_scale_info=self.total_chip_cap_info,
                chips_topology=chips_topology,
            )

            start = time.time()
            if is_random_assign:
                raise ValueError("During code implementation.")
            else:
                candidates_EPR_value = {}  # Record the cost of each candidate.
                total_assign_res = {}  # Record the mapping result of each candidate.
                total_par_res = {}

                pool_info = []
                for candidate_idx, chips_scale in chips_candidate_info.items():
                    # Partition the quantum circuit.
                    qucirc_partitioner = QuCircPartitioner()
                    par_res = qucirc_partitioner.run(self.total_qc, chips_scale, True)

                    new_par_res = self._repartiton(self.total_qc, par_res, chips_scale)

                    # Reset some global parameters.
                    self._reset_params()
                    self.par_res = self._adjust_par_res(
                        used_vir_qubits=self.used_vir_qubits,
                        ori_par_res=new_par_res,
                        chips_scale_info=chips_scale,
                    )

                    # Create the temporary index information of quantum chips.
                    self.candidate_chip_cap_info = chips_scale
                    self.chip_n_tmp_idx = self._create_chips_tmp_idx(
                        chips_cap_info=self.candidate_chip_cap_info
                    )

                    # Generate the information of quantum sub-circuits
                    (
                        self.qubit_blocks,
                        self.commu_qubits,
                        self.op_blocks,
                        self.remote_ops,
                    ) = create_sub_qcs_info(self.par_res, self.total_qc)

                    if is_global_calculate:
                        args = (
                            deepcopy(self.par_res),
                            self.chips,
                            self.total_qc,
                            deepcopy(self.qubit_blocks),
                            deepcopy(self.remote_ops),
                            deepcopy(self.chip_n_tmp_idx),
                            deepcopy(self.candidate_chip_cap_info),
                            self.chip_dist_matrix,
                            self.dist_matrix,
                            self.chips_nearest_commu_qubit,
                            self.fst_phase_iter,
                            self.sec_phase_iter,
                            FST_LST_LEN,
                            candidate_idx,
                        )
                        pool_info.append(args)
                    else:
                        args = (
                            deepcopy(self.par_res),
                            self.total_qc,
                            deepcopy(self.qubit_blocks),
                            deepcopy(self.remote_ops),
                            deepcopy(self.chip_n_tmp_idx),
                            deepcopy(self.candidate_chip_cap_info),
                            self.chip_dist_matrix,
                            self.dist_matrix,
                            self.chips_nearest_commu_qubit,
                            self.fst_phase_iter,
                            FST_LST_LEN,
                            candidate_idx,
                        )
                        pool_info.append(args)

                    total_par_res[candidate_idx] = deepcopy(self.par_res)

                pool = Pool()
                if is_global_calculate:
                    pool_res = pool.map(qg_assign_job_global_calculate, pool_info)
                    self.best_assign, self.best_par_res, self.best_mapping_res = (
                        self._select_best_alloca(pool_res, total_par_res)
                    )
                else:
                    pool_res = pool.map(qubit_group_assign_job, pool_info)
                    for index, global_alloc_cost, assign_res in pool_res:
                        candidates_EPR_value[index] = global_alloc_cost
                        total_assign_res[index] = assign_res

                    # Find the best candidate.
                    sorted_candidates_dist = sorted(
                        candidates_EPR_value.items(), key=lambda x: x[1], reverse=False
                    )

                    # The information about the best sub-circuit assignment.
                    best_candidate_idx = sorted_candidates_dist[0][0]
                    self.best_assign = total_assign_res[best_candidate_idx]
                    self.best_par_res = total_par_res[best_candidate_idx]
                    _, self.best_mapping_res = self._get_each_local_mapping(
                        self.best_assign,
                        self.best_par_res,
                        local_mapping_mode=local_mapping_mode,
                    )

                self.best_dis_num = self._analyse_par_res(self.best_par_res)

        else:
            raise ValueError(
                "There is only one number of quantum chips in a quantum chip network, so there is no need for distributed-oriented quantum program compilation."
            )

        end = time.time()
        self.run_time = end - start

        return self.best_mapping_res

    def _select_best_alloca(self, pool_res: List, total_par_res: Dict):
        total_mapping_cost = {}
        candidate_mapping_res = {}  # Record the mapping result of each candidate.

        candidates_cost = {}  # Record the cost of each candidate.
        total_assign_res = {}  # Record the mapping result of each candidate.
        for index, global_alloc_cost, assign_res, _ in pool_res:
            candidates_cost[index] = global_alloc_cost
            total_assign_res[index] = assign_res

        min_cost = min(candidates_cost.values())
        for index, assign_res in total_assign_res.items():
            if candidates_cost[index] == min_cost:
                local_mapper_cost, local_mapping_res = self._get_each_local_mapping(
                    assign_res, total_par_res[index]
                )
                total_mapping_cost[index] = local_mapper_cost + min_cost
                candidate_mapping_res[index] = local_mapping_res

                break

        sorted_assign_cost = sorted(
            total_mapping_cost.items(), key=lambda x: x[1], reverse=False
        )

        best_candidate_index = sorted_assign_cost[0][0]
        best_mapping_res = candidate_mapping_res[best_candidate_index]

        return (
            total_assign_res[best_candidate_index],
            total_par_res[best_candidate_index],
            best_mapping_res,
        )

    def _get_each_local_mapping(
        self,
        alloc_info: bidict,
        par_res: Dict[Qubit, int],
        local_mapping_mode: str = "hsa",
    ) -> bidict:
        """
        Get the local optimal qubits mapping results and the corresponding Multiplicative fidelity.
        """
        init_mapping_res = bidict()
        tmp_init_mapping_info = {}

        pool_res = []
        total_cost = 0
        if local_mapping_mode == "hta":
            (
                qubit_blocks,
                commu_qubits,
                op_blocks,
                _,
            ) = create_sub_qcs_info(par_res, self.total_qc)

            for qubit_group_idx, chip_idx in alloc_info.items():
                pool_info = []

                # Get the qubit topology of the quantum chip
                qubits_topology = None
                for chip in self.chips:
                    if chip.index == chip_idx:
                        qubits_topology = chip.get_chip_topology()
                        break

                args = (
                    deepcopy(par_res),
                    deepcopy(alloc_info),
                    deepcopy(qubit_blocks[qubit_group_idx]),
                    deepcopy(commu_qubits),
                    deepcopy(op_blocks[qubit_group_idx]),
                    deepcopy(self.chips_nearest_commu_qubit),
                    deepcopy(qubits_topology),
                    self.dist_matrix,
                    self.sec_phase_iter,
                    SEC_LST_LEN,
                    True,
                    False,
                    qubit_group_idx,
                )
                pool_info.append(args)
            pool = Pool()
            pool_res = pool.map(single_chip_map_job, pool_info)

            for index, cost_value, local_mapping_res in pool_res:
                tmp_init_mapping_info[index] = local_mapping_res
                total_cost += cost_value

            for _, mapping_info in tmp_init_mapping_info.items():
                for vir_qubit, phy_qubit in mapping_info.items():
                    init_mapping_res[vir_qubit] = phy_qubit
        elif local_mapping_mode == "hsa":
            total_cost = 0
            dist_matrix_tolist = self.dist_matrix.tolist()

            local_mappers = IntraProcessMapper()
            local_mapping_res = local_mappers.run(
                total_circuit=self.total_qc,
                par_res=par_res,
                alloc_info=alloc_info,
                dist_matrix_tolist=dist_matrix_tolist,
                total_coupling_map=self.total_coupling_map,
                ep_coupling_map=self.ep_coupling_map,
                ep_phy_qubits=self.ep_physical_qubits,
                chips_nearest_commu_qubits=self.nearest_phy_commu_qubits,
                iter_num=5,
                lookahead_ability=20,
                intra_chip_all2all=self.intra_chip_all2all,
            )

        return total_cost, local_mapping_res

    def _obtain_coupling_map(self, chips, remote_connections):
        ep_physical_qubits = {}
        total_coupling_map = None
        dict_coupling_map = {}

        # Used for the local qubit mapping based on the SABRE algorithm.
        total_coup_info = []
        for chip in chips:
            ep_physical_qubits[chip.index] = chip.qubits_index

            local_coup_info = []
            for local_coup in chip.couplings:
                idx_pair = (local_coup[0], local_coup[1])
                local_coup_info.append(idx_pair)
                total_coup_info.append(idx_pair)

            local_coup_map = CouplingMap(local_coup_info)
            local_coup_map.make_symmetric()
            dict_coupling_map[chip.index] = deepcopy(local_coup_map)

        for chip_connect in remote_connections:
            qubit_pair = chip_connect.qubit_pair
            qubit_idx_pair = (qubit_pair[0].index, qubit_pair[1].index)
            total_coup_info.append(qubit_idx_pair)

        total_coupling_map = CouplingMap(total_coup_info)
        total_coupling_map.make_symmetric()

        return ep_physical_qubits, total_coupling_map, dict_coupling_map

    def _analyse_par_res(self, par_res):
        analyse_res = {}

        for _, v in par_res.items():
            if v in analyse_res:
                analyse_res[v] += 1
            else:
                analyse_res[v] = 1

        return analyse_res

    def _analyze_chip_candidates(
        self,
        num_virtual_qubits: int,
        chip_scale_info: Dict[int, int],
        chips_topology: nx.Graph,
        work_mode: str = "min",
    ) -> Dict[int, Dict[int, int]]:
        """
        Analyze all the chip cluster candidates (Find the smallest collection of quantum chips that meet the needs of virtual qubits).
        Note: We only consider the scenario with the minimum requirement of physical qubits.
        """
        chip_clus_candidates = {}

        chips_idx = list(chip_scale_info.keys())
        if work_mode == "max":
            # Get the all subset of the chip index list.
            all_subsets = []

            for num in range(len(chips_idx)):
                for subset in combinations(chips_idx, num + 1):
                    all_subsets.append(subset)

            # Analyze all chip clusters and get the candidate chip clusters.
            subset_idx = 0
            for subset in all_subsets:
                tmp_data = {}
                num_phy_qubits = 0

                for chip_idx in subset:
                    num_phy_qubits += chip_scale_info[chip_idx]
                    tmp_data[chip_idx] = chip_scale_info[chip_idx]

                # Judge if the number of physical qubits is enough.
                if num_phy_qubits >= num_virtual_qubits:
                    chip_clus_candidates[subset_idx] = tmp_data
                    subset_idx += 1
        elif work_mode == "min":
            all_subsets = []

            subset_idx = 0
            for num in range(len(chips_idx)):
                for subset in combinations(chips_idx, num + 1):
                    tmp_data = {}
                    num_phy_qubits = 0

                    for idx in subset:
                        num_phy_qubits += chip_scale_info[idx]
                        tmp_data[idx] = chip_scale_info[idx]

                    if num_phy_qubits >= num_virtual_qubits:
                        if self._is_subset_connected(list(subset), chips_topology):
                            all_subsets.append(subset)
                            chip_clus_candidates[subset_idx] = tmp_data
                            subset_idx += 1

                if len(all_subsets) > 0:
                    break

        return chip_clus_candidates

    def _repartiton(
        self,
        quantum_circuit: QuantumCircuit,
        par_res: Dict[Qubit, int],
        chips_scale_info: Dict[int, int],
    ):
        """
        Judge if the partition result is usable.
        """
        # Count the size of each partitioned block.
        blocks_info = {}
        blocks_size = {}
        for qubit_idx, block_id in par_res.items():
            if block_id not in blocks_info:
                blocks_info[block_id] = [qubit_idx]
                blocks_size[block_id] = 1
            else:
                blocks_info[block_id].append(qubit_idx)
                blocks_size[block_id] += 1

        # Judge if the size of each partitioned block is less than the size of the chip.
        sorted_chips_scale = sorted(
            chips_scale_info.items(), key=lambda x: x[1], reverse=True
        )
        sorted_blocks_size = sorted(
            blocks_size.items(), key=lambda x: x[1], reverse=True
        )
        new_blocks_info = {
            ele[0]: deepcopy(blocks_info[ele[0]]) for ele in sorted_blocks_size
        }

        for i in range(len(blocks_info) - 1):
            block_id = sorted_blocks_size[i][0]
            qubits = new_blocks_info[block_id]
            chip_scale = sorted_chips_scale[i][1]

            next_block_id = sorted_blocks_size[i + 1][0]

            if len(qubits) > chip_scale:
                qubits_degree = {qubit: 0 for qubit in qubits}
                for insn in quantum_circuit:
                    if insn.operation.name == "cx" or insn.operation.name == "cz":
                        fst_qubit, sec_qubit = insn.qubits[0], insn.qubits[1]
                        if fst_qubit in qubits and sec_qubit in qubits:
                            qubits_degree[fst_qubit] += 1
                            qubits_degree[sec_qubit] += 1

                sorted_qubits = sorted(
                    qubits_degree.items(), key=lambda x: x[1], reverse=False
                )
                num_reallocate_qubits = len(qubits) - chip_scale

                for j in range(num_reallocate_qubits):
                    qubits.remove(sorted_qubits[j][0])
                    new_blocks_info[next_block_id].append(sorted_qubits[j][0])

        new_par_res = {}
        for block_id, qubits in new_blocks_info.items():
            for qubit in qubits:
                new_par_res[qubit] = block_id

        return new_par_res

    def _reset_params(self) -> None:
        """
        Reset the global parameters.
        """
        self.chip_n_tmp_idx = None
        self.candidate_chip_cap_info = None

        self.qubit_blocks = None
        self.commu_qubits = None
        self.op_blocks = None
        self.remote_ops = None

    def _create_aux_robust_info(self):
        """
        Create the auxiliary robustness information that available for estimate the robustness of remote communication.
        """
        aux_robust_info = {}

        for chip in self.chips:
            qubits_topology = chip.get_chip_topology()

            phy_commu_qubits = self.each_chip_commu_qubits[chip.index]
            for qubit in phy_commu_qubits:
                tmp_fid = 1.0
                num_neighbors = 0
                neighbors = qubits_topology.neighbors(qubit)
                for neighbor in neighbors:
                    num_neighbors += 1
                    coupling_info = qubits_topology.get_edge_data(qubit, neighbor)
                    tmp_fid *= coupling_info["fidelity"]

                aux_rb_value = pow(pow(tmp_fid, 1 / num_neighbors), 3)
                aux_robust_info[qubit] = aux_rb_value

        return aux_robust_info


    def _obtain_nearest_commu_qubits(
        self,
    ) -> Dict[Tuple[int], Tuple[int]]:
        """
        Calculate the distance value of the most nearest path between two quantum chips, and record the information of communication physical qubits at both ends of the path.
        """
        chips_idx = list(self.total_chip_cap_info.keys())
        chips_nearest_commu_qubit = {}

        for chip_idx_pair in itertools.combinations(chips_idx, 2):
            fst_chip_idx = chip_idx_pair[0]
            sec_chip_idx = chip_idx_pair[1]
            chips_nearest_commu_qubit[chip_idx_pair] = []

            cost_info = {}
            for commu_qubit_1 in self.each_chip_commu_qubits[fst_chip_idx]:
                for commu_qubit_2 in self.each_chip_commu_qubits[sec_chip_idx]:
                    cost_info[(commu_qubit_1, commu_qubit_2)] = self.dist_matrix[
                        commu_qubit_1
                    ][commu_qubit_2]

            min_cost = min(cost_info.values())
            for commu_qubit_pair, cost in cost_info.items():
                if cost == min_cost:
                    chips_nearest_commu_qubit[chip_idx_pair].append(commu_qubit_pair)

        return chips_nearest_commu_qubit

    def _create_commu_rout_info(
        self,
    ) -> Dict[Tuple[int], Tuple[int]]:
        """
        Calculate the distance value of the most nearest path between two quantum chips, and record the information of communication physical qubits at both ends of the path.
        """
        chips_idx = list(self.total_chip_cap_info.keys())
        chips_nearest_commu_qubit = {}

        for chip_idx_pair in itertools.combinations(chips_idx, 2):
            fst_chip_idx = chip_idx_pair[0]
            sec_chip_idx = chip_idx_pair[1]

            min_cost = np.inf
            nearest_commu_qubit = None
            for commu_qubit_1 in self.each_chip_commu_qubits[fst_chip_idx]:
                for commu_qubit_2 in self.each_chip_commu_qubits[sec_chip_idx]:
                    current_cost = self.dist_matrix[commu_qubit_1][commu_qubit_2]

                    if current_cost < min_cost:
                        min_cost = current_cost
                        nearest_commu_qubit = (commu_qubit_1, commu_qubit_2)

            chips_nearest_commu_qubit[chip_idx_pair] = nearest_commu_qubit

        return chips_nearest_commu_qubit

    def _create_chips_tmp_idx(self, chips_cap_info: Dict[int, int]) -> bidict:
        """
        Create the temporary index information of quantum chips.
        """
        chip_n_tmp_idx = bidict()
        enum_chip_idx = enumerate(list(chips_cap_info.keys()))
        for tmp_idx, chip_idx in enum_chip_idx:
            chip_n_tmp_idx[chip_idx] = tmp_idx

        return chip_n_tmp_idx

    def _count_used_qubits(self, quantum_circuit: QuantumCircuit) -> List[Qubit]:
        """
        Get the list of used qubits.
        """
        used_qubits = []
        for insn in quantum_circuit:
            for qubit in insn.qubits:
                if qubit not in used_qubits:
                    used_qubits.append(qubit)
        return used_qubits

    def _adjust_par_res(
        self,
        used_vir_qubits: List[Qubit],
        ori_par_res: Dict[Qubit, int],
        chips_scale_info: Dict[int, int],
    ) -> Dict:
        """
        Since some qubits are only acted by single-qubit operations, the qubit group result needs to be adjusted to add these qubits.
        """
        qubits_in_par_res = list(ori_par_res.keys())

        if len(used_vir_qubits) > len(qubits_in_par_res):
            new_par_res = deepcopy(ori_par_res)

            # Get the qubits that are not no in the partition result.
            unsolved_qubits = []
            for qubit in used_vir_qubits:
                if qubit not in qubits_in_par_res:
                    unsolved_qubits.append(qubit)

            # Calculate the number of qubits in each qubits group of origin partiton result.
            par_info = {}
            for _, chip_idx in ori_par_res.items():
                if chip_idx in par_info:
                    par_info[chip_idx] += 1
                else:
                    par_info[chip_idx] = 1

            # Add the qubits that are not in the partition result to the qubits group.
            chips_list = list(chips_scale_info.keys())
            for chip_idx in chips_list:
                while unsolved_qubits:
                    if (
                        chip_idx in par_info
                        and par_info[chip_idx] < chips_scale_info[chip_idx]
                    ):
                        new_par_res[unsolved_qubits[0]] = chip_idx
                        unsolved_qubits.pop(0)
                        par_info[chip_idx] += 1
                    elif chip_idx not in par_info:
                        new_par_res[unsolved_qubits[0]] = chip_idx
                        unsolved_qubits.pop(0)
                        par_info[chip_idx] = 1
                    else:
                        break

                if not unsolved_qubits:
                    break

            return new_par_res
        elif len(used_vir_qubits) == len(qubits_in_par_res):
            return ori_par_res

    def _is_subset_connected(self, chip_list, chips_topology):
        """Determine whether the list of nodes in the undirected graph G is connected"""
        if not chip_list:
            return False  

        # Select a starting point in the nodes
        start_node = next(iter(chip_list))

        visited = set()
        queue = [start_node]

        while queue:
            node = queue.pop(0)
            if node in chip_list and node not in visited:
                visited.add(node)
                queue.extend(set(chips_topology.neighbors(node)) - visited)

        # If all the points in the nodes are accessed, they are connected
        return visited >= set(chip_list)

class GlobalAllocation:
    """
    Finding the optimal assignment relationship between qubit groups and quantum chips.
    """

    def __init__(self) -> None:
        self.par_res = None

        # The information of quantum circuit
        self.total_qc = None
        self.remote_ops = None
        self.qubit_blocks = None

        # The information of quantum chip network
        self.chips_info = None
        self.chip_n_tmp_idx = None
        self.candidate_chip_cap_info = None

        self.chip_dist_matrix = None
        self.cost_matrix = None

        self.global_calculate = False

    def search_random_qubit_group_alloc(
        self,
        qubit_blocks: Dict[int, List[int]],
        chip_n_tmp_idx: bidict,
        candidate_chip_cap_info: Dict[int, int],
    ) -> bidict:
        self.qubit_blocks = qubit_blocks
        self.chip_n_tmp_idx = chip_n_tmp_idx
        self.candidate_chip_cap_info = candidate_chip_cap_info

        tmp_alloc_res = self._init_assign()
        final_alloc_res = self._get_assign_info(tmp_alloc_res, self.chip_n_tmp_idx)

        return final_alloc_res

    def search_opt_qubit_group_alloc(
        self,
        par_res: Dict,
        quantum_circuit: QuantumCircuit,
        qubit_blocks: Dict[int, List[int]],
        remote_ops: List[CircuitInstruction],
        chip_n_tmp_idx: bidict,
        candidate_chip_cap_info: Dict[int, int],
        chip_dist_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        chips_nearest_commu_qubit: Dict[Tuple[int, int], Tuple[int, int]],
        num_iter=100,
        lst_len=20,
    ) -> Tuple[float, bidict]:
        """
        Search the near-optimal quantum sub-circuit assignment result.
        And return the qubits initial mapping result of origin quantum circuit.
        """
        self._update_global_para(
            par_res,
            quantum_circuit,
            qubit_blocks,
            remote_ops,
            chip_n_tmp_idx,
            candidate_chip_cap_info,
            chip_dist_matrix,
        )
        self.cost_matrix = cost_matrix
        self.chips_nearest_commu_qubit = chips_nearest_commu_qubit

        best_assign_res = None
        mini_cost = None

        num_candidate_chips = len(self.chip_n_tmp_idx)
        N = int(num_candidate_chips * (num_candidate_chips - 1) / 2)
        neighbors = np.zeros((N, num_candidate_chips + 2), dtype=int)

        # get the assignment information
        current_assign = self._init_assign()
        mini_cost, best_assign_res, _ = self._estimate_commu_dist(current_assign)

        tabu_list = []
        frequency = {}
        while num_iter > 0:
            # Update the neighbors of current assignment
            self._assign_swap_move(num_candidate_chips, current_assign, neighbors)

            # Sorts the elements in the neighbors according to the cost
            # cost = np.zeros(N)
            cost = np.ones(len(neighbors)) * np.inf
            for i in range(len(neighbors)):
                current_assign = neighbors[i, :-2].tolist()

                is_correct_assign = self._judge_correct_assign(current_assign)
                if is_correct_assign:
                    # Evaluate the candidate assignment
                    cost[i], _, _ = self._estimate_commu_dist(current_assign)
            rank = np.argsort(cost)
            neighbors = neighbors[rank]

            for j in range(N):
                current_assign = neighbors[j, :-2].tolist()
                is_correct_assign = self._judge_correct_assign(current_assign)

                if is_correct_assign:
                    not_in_tabu_list = self._not_in_tabu_list(
                        deepcopy(neighbors[j, -2:]), tabu_list
                    )

                    if not_in_tabu_list:
                        tuple_cur_assign = tuple(current_assign)

                        # update the tabu list
                        tabu_list.append(neighbors[j, -2:].tolist())
                        if len(tabu_list) > lst_len - 1:
                            tabu_list.pop(0)

                        cur_cost, cur_assign_res, _ = self._estimate_commu_dist(
                            current_assign
                        )
                        if not tuple_cur_assign in frequency.keys():
                            frequency[tuple_cur_assign] = FST_PHASE_DISTURBED_VALUE

                            if cur_cost < mini_cost:
                                mini_cost = cur_cost
                                best_assign_res = cur_assign_res
                        else:
                            cur_cost += frequency[tuple_cur_assign]
                            frequency[tuple_cur_assign] += FST_PHASE_DISTURBED_VALUE

                            if cur_cost < mini_cost:
                                mini_cost = cur_cost
                                best_assign_res = cur_assign_res
                        break
                    else:
                        cur_cost, cur_assign_res, _ = self._estimate_commu_dist(
                            current_assign
                        )

                        if cur_cost < mini_cost:
                            tuple_cur_assign = tuple(current_assign)

                            # Put the current assignment into the first location of tabu list
                            swap_info = neighbors[j, -2:].tolist()
                            rev_swap_info = deepcopy(swap_info)
                            rev_swap_info[0], rev_swap_info[1] = (
                                rev_swap_info[1],
                                rev_swap_info[0],
                            )

                            if swap_info in tabu_list:
                                tabu_list.insert(
                                    0, tabu_list.pop(tabu_list.index(swap_info))
                                )
                            else:
                                if rev_swap_info in tabu_list:
                                    tabu_list.insert(
                                        0, tabu_list.pop(tabu_list.index(rev_swap_info))
                                    )

                            if len(tabu_list) > lst_len - 1:
                                tabu_list.pop(0)

                            if not tuple_cur_assign in frequency.keys():
                                frequency[tuple_cur_assign] = FST_PHASE_DISTURBED_VALUE
                                mini_cost = cur_cost
                                best_assign_res = cur_assign_res
                            else:
                                cur_cost += frequency[tuple_cur_assign]
                                frequency[tuple_cur_assign] += FST_PHASE_DISTURBED_VALUE

                                if cur_cost < mini_cost:
                                    mini_cost = cur_cost
                                    best_assign_res = cur_assign_res

            num_iter -= 1

        return mini_cost, best_assign_res

    def global_opt_qubit_group_alloc(
        self,
        par_res: Dict,
        quantum_chips: set,
        quantum_circuit: QuantumCircuit,
        qubit_blocks: Dict[int, List[int]],
        remote_ops: List[CircuitInstruction],
        chip_n_tmp_idx: bidict,
        candidate_chip_cap_info: Dict[int, int],
        chip_dist_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        chips_nearest_commu_qubit: Dict[Tuple[int, int], Tuple[int, int]],
        fst_num_iter=100,
        sec_num_iter=200,
        lst_len=20,
    ) -> Tuple[float, bidict, bidict]:
        """
        Search the near-optimal quantum sub-circuit assignment result.
        And return the qubits initial mapping result of origin quantum circuit.
        """
        self._update_global_para(
            par_res,
            quantum_circuit,
            qubit_blocks,
            remote_ops,
            chip_n_tmp_idx,
            candidate_chip_cap_info,
            chip_dist_matrix,
        )
        self.chips_info = quantum_chips
        self.cost_matrix = cost_matrix
        self.global_calculate = True
        self.sec_num_iter = sec_num_iter
        self.chips_nearest_commu_qubit = chips_nearest_commu_qubit

        best_assign_res = None
        best_mapping_res = None
        mini_cost = None

        num_candidate_chips = len(self.chip_n_tmp_idx)
        N = int(num_candidate_chips * (num_candidate_chips - 1) / 2)
        neighbors = np.zeros((N, num_candidate_chips + 2), dtype=int)

        # get the assignment information
        current_assign = self._init_assign()
        mini_cost, best_assign_res, best_mapping_res = self._estimate_commu_dist(
            current_assign
        )

        tabu_list = []
        frequency = {}
        while fst_num_iter > 0:
            # Update the neighbors of current assignment
            self._assign_swap_move(num_candidate_chips, current_assign, neighbors)

            # Sorts the elements in the neighbors according to the cost
            # cost = np.zeros(N)
            cost = np.ones(len(neighbors)) * np.inf
            for i in range(len(neighbors)):
                current_assign = neighbors[i, :-2].tolist()

                is_correct_assign = self._judge_correct_assign(current_assign)
                if is_correct_assign:
                    # Evaluate the candidate assignment
                    cost[i], _, _ = self._estimate_commu_dist(current_assign)
            rank = np.argsort(cost)
            neighbors = neighbors[rank]

            for j in range(N):
                current_assign = neighbors[j, :-2].tolist()
                is_correct_assign = self._judge_correct_assign(current_assign)

                if is_correct_assign:
                    not_in_tabu_list = self._not_in_tabu_list(
                        deepcopy(neighbors[j, -2:]), tabu_list
                    )

                    if not_in_tabu_list:
                        tuple_cur_assign = tuple(current_assign)

                        # update the tabu list
                        tabu_list.append(neighbors[j, -2:].tolist())
                        if len(tabu_list) > lst_len - 1:
                            tabu_list.pop(0)

                        (
                            cur_EPRs_value,
                            cur_assign_res,
                            cur_mapping_res,
                        ) = self._estimate_commu_dist(current_assign)
                        if not tuple_cur_assign in frequency.keys():
                            frequency[tuple_cur_assign] = FST_PHASE_DISTURBED_VALUE

                            if cur_EPRs_value < mini_cost:
                                mini_cost = cur_EPRs_value
                                best_assign_res = cur_assign_res
                                best_mapping_res = cur_mapping_res
                        else:
                            cur_EPRs_value += frequency[tuple_cur_assign]
                            frequency[tuple_cur_assign] += FST_PHASE_DISTURBED_VALUE

                            if cur_EPRs_value < mini_cost:
                                mini_cost = cur_EPRs_value
                                best_assign_res = cur_assign_res
                                best_mapping_res = cur_mapping_res
                        break
                    else:
                        (
                            cur_EPRs_value,
                            cur_assign_res,
                            cur_mapping_res,
                        ) = self._estimate_commu_dist(current_assign)

                        if cur_EPRs_value < mini_cost:
                            tuple_cur_assign = tuple(current_assign)

                            # Put the current assignment into the first location of tabu list
                            swap_info = neighbors[j, -2:].tolist()
                            rev_swap_info = deepcopy(swap_info)
                            rev_swap_info[0], rev_swap_info[1] = (
                                rev_swap_info[1],
                                rev_swap_info[0],
                            )

                            if swap_info in tabu_list:
                                tabu_list.insert(
                                    0, tabu_list.pop(tabu_list.index(swap_info))
                                )
                            else:
                                if rev_swap_info in tabu_list:
                                    tabu_list.insert(
                                        0, tabu_list.pop(tabu_list.index(rev_swap_info))
                                    )

                            if len(tabu_list) > lst_len - 1:
                                tabu_list.pop(0)

                            if not tuple_cur_assign in frequency.keys():
                                frequency[tuple_cur_assign] = FST_PHASE_DISTURBED_VALUE
                                mini_cost = cur_EPRs_value
                                best_assign_res = cur_assign_res
                                best_mapping_res = cur_mapping_res
                            else:
                                cur_EPRs_value += frequency[tuple_cur_assign]
                                frequency[tuple_cur_assign] += FST_PHASE_DISTURBED_VALUE

                                if cur_EPRs_value < mini_cost:
                                    mini_cost = cur_EPRs_value
                                    best_assign_res = cur_assign_res
                                    best_mapping_res = best_mapping_res

            fst_num_iter -= 1

        return mini_cost, best_assign_res, best_mapping_res

    def _update_global_para(
        self,
        par_res: Dict,
        quantum_circuit: QuantumCircuit,
        qubit_blocks: Dict[int, List[int]],
        remote_ops: List[CircuitInstruction],
        chip_n_tmp_idx: bidict,
        candidate_chip_cap_info: Dict[int, int],
        chip_dist_matrix: np.ndarray,
    ):
        self.par_res = par_res

        # The information of quantum circuit
        self.total_qc = quantum_circuit
        self.remote_ops = remote_ops
        self.qubit_blocks = qubit_blocks

        # The information of quantum chip network
        self.chip_n_tmp_idx = chip_n_tmp_idx
        self.candidate_chip_cap_info = candidate_chip_cap_info

        self.chip_dist_matrix = chip_dist_matrix

    def _init_assign(self) -> List[int]:
        """
        Initialize the assignment of sub-circuits to quantum chips.
        """
        init_assign = [-(i + 1) for i in range(len(self.candidate_chip_cap_info))]

        # They are sorted according to the size of the sub-circuit and capacity of quantum chip, respectively.
        chips_cap_sorted = sorted(
            self.candidate_chip_cap_info.items(), key=lambda x: x[1], reverse=True
        )
        blocks_size_sorted = sorted(
            self.qubit_blocks.items(), key=lambda x: len(x[1]), reverse=True
        )

        num_assigned_circ = 0
        for block_idx, qubit_block in blocks_size_sorted:
            for chip_idx, chip_cap in chips_cap_sorted:
                if chip_cap >= len(qubit_block):
                    init_assign[self.chip_n_tmp_idx[chip_idx]] = block_idx
                    chips_cap_sorted.remove((chip_idx, chip_cap))

                    num_assigned_circ += 1
                    break

        if num_assigned_circ != len(self.qubit_blocks):
            raise ValueError(
                "Cannot find a correct initial assignment. Because the capacity of the quantum chip is smaller than the size of the quantum sub-circuit"
            )

        return init_assign

    def _estimate_commu_dist(
        self, current_assign: List[int]
    ) -> Tuple[float, bidict, bidict]:
        """
        Estimate the total communication distance value.
        """
        estimated_value = 0
        assign_res = self._get_assign_info(current_assign, self.chip_n_tmp_idx)
        mapping_res = bidict()

        for insn in self.remote_ops:
            fst_chip_idx = assign_res[self.par_res[insn[1][0]]]
            sec_chip_idx = assign_res[self.par_res[insn[1][1]]]
            dist_value = self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]

            # Get the chip index pair and the related communication qubits information.
            chip_idx_pair = (fst_chip_idx, sec_chip_idx)
            rev_chip_idx_pair = (sec_chip_idx, fst_chip_idx)

            commu_qubit_pair = None
            if chip_idx_pair in self.chips_nearest_commu_qubit:
                commu_qubit_pair = self.chips_nearest_commu_qubit[chip_idx_pair]
            if rev_chip_idx_pair in self.chips_nearest_commu_qubit:
                commu_qubit_pair = self.chips_nearest_commu_qubit[rev_chip_idx_pair]

            if dist_value > 0:
                tmp_value = (
                    self.cost_matrix[commu_qubit_pair[0]][commu_qubit_pair[1]]
                    + (2 * dist_value - 1) * EPR_COST
                )

                estimated_value += tmp_value

        return estimated_value, assign_res, mapping_res

    def _assign_swap_move(
        self, num_chips: int, current_assign: List[int], neighbors: np.ndarray
    ) -> None:
        """
        Generate the current neighbors of the current solution.
        """
        neighbor_idx = -1
        for i in range(num_chips):
            for j in range(num_chips):
                if i < j:
                    neighbor_idx += 1
                    current_assign[j], current_assign[i] = (
                        current_assign[i],
                        current_assign[j],
                    )
                    neighbors[neighbor_idx, :-2] = current_assign
                    neighbors[neighbor_idx, -2:] = [
                        current_assign[i],
                        current_assign[j],
                    ]
                    current_assign[i], current_assign[j] = (
                        current_assign[j],
                        current_assign[i],
                    )

    def _judge_correct_assign(self, current_assign: List[int]) -> bool:
        """
        Judge whether the subcircuit assignment is correct, i.e. the size of subcircuit can not less than the capacity of quantum chip.
        """
        is_correct = True

        index = 0
        for ele in current_assign:
            if ele >= 0:
                if (
                    len(self.qubit_blocks[ele])
                    > self.candidate_chip_cap_info[self.chip_n_tmp_idx.inverse[index]]
                ):
                    is_correct = False
                    break
            index += 1

        return is_correct

    def _not_in_tabu_list(self, assignment_tag, tabu_list) -> bool:
        """
        Check if the assignment is in the tabu list.
        """
        not_found = False
        if not assignment_tag.tolist() in tabu_list:
            assignment_tag[0], assignment_tag[1] = assignment_tag[1], assignment_tag[0]
            if not assignment_tag.tolist() in tabu_list:
                not_found = True

        return not_found

    def _get_assign_info(
        self, current_assign: List[int], chip_n_tmp_idx: bidict
    ) -> bidict:
        """
        Get the mapping information of quantum sub-circuits and quantum chips.
        """
        assign_res = bidict()

        for index in range(len(current_assign)):
            if current_assign[index] >= 0:
                assign_res[current_assign[index]] = chip_n_tmp_idx.inverse[index]

        return assign_res

    def _local_search(
        self, assign_info: bidict, par_res: Dict[Qubit, int]
    ) -> Dict[int, bidict]:
        """
        Get the local optimal qubits mapping results and the corresponding Multiplicative fidelity.
        """
        total_local_value = 0
        init_mapping_res = bidict()
        tmp_init_mapping_info = {}

        (
            qubit_blocks,
            commu_qubits,
            op_blocks,
            remote_ops,
        ) = create_sub_qcs_info(par_res, self.total_qc)

        for block_idx, chip_idx in assign_info.items():
            # Get the qubit topology of the quantum chip
            qubits_topology = None
            for chip in self.chips_info:
                if chip.index == chip_idx:
                    qubits_topology = chip.get_chip_topology()
                    break

            local_mapper = LocalInitMapper()
            cost_value, local_mapping_res = local_mapper.obtain_local_init_mapping(
                par_res=par_res,
                assign_res=assign_info,
                vir_qubit_block=qubit_blocks[block_idx],
                commu_vir_qubits=commu_qubits,
                op_block=op_blocks[block_idx],
                chips_commu_info=self.chips_nearest_commu_qubit,
                qubits_topology=qubits_topology,
                cost_matrix=self.cost_matrix,
                num_iter=self.sec_num_iter,
                lst_len=SEC_LST_LEN,
                is_return_dist=True,
                print_cost_info=False,
            )

            total_local_value += cost_value
            tmp_init_mapping_info[block_idx] = local_mapping_res

        for _, mapping_info in tmp_init_mapping_info.items():
            for vir_qubit, phy_qubit in mapping_info.items():
                init_mapping_res[vir_qubit] = phy_qubit

        return total_local_value, init_mapping_res


class LocalInitMapper:
    """
    Using Tabu search heuristic algorithm to find an initial mapping of a quantum program in the local quantum chip.
    """

    def __init__(self) -> None:
        self.par_res = None
        self.assign_res = None

        self.flow_matrix = None
        self.cost_matrix = None

        self.local_2q_ops = None
        self.rm_2q_ops = None

        self.vir_qubits = None
        self.commu_vir_qubits = None
        self.chips_commu_info = None

        self.vir_qubit_n_tmp_idx = bidict()
        self.phy_qubit_n_tmp_idx = bidict()

        self.best_sol = None
        self.min_dist = None

    def obtain_local_init_mapping(
        self,
        par_res: Dict[int, int],
        assign_res: bidict,
        vir_qubit_block: List[Qubit],
        commu_vir_qubits: List[Qubit],
        op_block: List[CircuitInstruction],
        chips_commu_info: Dict[Tuple[int, int], Tuple[int, int]],
        qubits_topology: nx.Graph,
        cost_matrix: np.ndarray,
        num_iter=200,
        lst_len=20,
        is_return_dist=False,
        print_cost_info=False,
    ) -> bidict:
        """
        Starts the search for an initial mapping.
        """
        mapping_result = bidict()

        # Initialize the global variables.
        self.par_res = par_res
        self.assign_res = assign_res
        self.chips_commu_info = chips_commu_info
        self.vir_qubits = vir_qubit_block
        self.commu_vir_qubits = commu_vir_qubits

        # Classify the quantum operations.
        self.rm_2q_ops, self.local_2q_ops = self._classify_ops(op_block)

        # Generate the temporary index of virtual qubits.
        for tmp_idx, qubit in enumerate(vir_qubit_block):
            self.vir_qubit_n_tmp_idx[qubit] = tmp_idx
        num_vir_qubits = len(self.vir_qubit_n_tmp_idx)

        # Generate the temporary index of graph nodes.
        nodes_idx = list(qubits_topology.nodes)
        for tmp_idx, node_idx in enumerate(nodes_idx):
            self.phy_qubit_n_tmp_idx[node_idx] = tmp_idx

        # TODO: This judgment here may be directly deleted later.
        num_phy_qubits = qubits_topology.number_of_nodes()
        if num_vir_qubits > num_phy_qubits:
            raise ValueError("More virtual qubits exist than physical qubits.")

        self.flow_matrix = self._create_flow_matrix()
        if cost_matrix is not None:
            self.cost_matrix = cost_matrix
        else:
            raise ValueError("The estimated routing cost matrix must be given.")

        # Build a current solution randomly.
        current_sol = self._create_init_sol(num_phy_qubits, num_vir_qubits)
        self.best_sol = current_sol
        self.min_dist = self._estimate_dist_value(self.best_sol)

        # Print the distance value of the best solution.
        if print_cost_info:
            print("The cost of solution before Tabu search is: ", self.min_dist)

        # Initialize the tabu list.
        N = int(num_phy_qubits * (num_phy_qubits - 1) / 2)
        neighbors = np.zeros((N, num_phy_qubits + 2), dtype=int)
        tabu_list = []
        frequency = {}

        while num_iter > 0:
            # Update the neighbors of the current solution.
            self._swap_move(num_phy_qubits, current_sol, neighbors)

            # Sorts the elements in the neighbors.
            cost = np.ones(N) * np.inf  # Holds the cost of neighbors.
            for i in range(N):
                tmp_sol = neighbors[i, :-2].tolist()
                cost[i] = self._estimate_dist_value(tmp_sol)
            rank = np.argsort(cost)  # Sorted index based on cost.
            neighbors = neighbors[rank]

            for j in range(N):
                not_in_tabu_list = self._not_in_tabu_list(
                    deepcopy(neighbors[j, -2:]), tabu_list
                )

                if not_in_tabu_list:
                    current_sol = neighbors[j, :-2].tolist()
                    tuple_current_sol = tuple(current_sol)

                    # Upadate the tabu list.
                    tabu_list.append(neighbors[j, -2:].tolist())

                    if len(tabu_list) > lst_len - 1:
                        tabu_list.pop(0)  # Pop the first element of  tabu list.

                    # Frequency based
                    if not tuple_current_sol in frequency.keys():
                        frequency[tuple_current_sol] = SEC_PHASE_DISTURBED_VALUE

                        current_dist_value = self._estimate_dist_value(current_sol)
                        if current_dist_value < self.min_dist:
                            self.best_sol = current_sol
                            self.min_dist = current_dist_value
                    else:
                        tmp_dist_value = self._estimate_dist_value(current_sol)
                        current_dist_value = (
                            tmp_dist_value + frequency[tuple_current_sol]
                        )  # penalize by frequency

                        # Update the frequency.
                        frequency[tuple_current_sol] += SEC_PHASE_DISTURBED_VALUE

                        if current_dist_value < self.min_dist:
                            self.best_sol = current_sol
                            self.min_dist = current_dist_value
                    break
                else:
                    current_sol = neighbors[j, :-2].tolist()
                    current_dist_value = self._estimate_dist_value(current_sol)

                    if current_dist_value < self.min_dist:
                        tuple_current_sol = tuple(current_sol)

                        # Put the current solution to the first location of tabu list.
                        swap_info = neighbors[j, -2:].tolist()
                        rev_swap_info = deepcopy(swap_info)
                        rev_swap_info[0], rev_swap_info[1] = (
                            rev_swap_info[1],
                            rev_swap_info[0],
                        )

                        if swap_info in tabu_list:
                            tabu_list.insert(
                                0, tabu_list.pop(tabu_list.index(swap_info))
                            )
                        else:
                            if rev_swap_info in tabu_list:
                                tabu_list.insert(
                                    0, tabu_list.pop(tabu_list.index(rev_swap_info))
                                )

                        if len(tabu_list) > lst_len - 1:
                            tabu_list.pop(0)

                        if not tuple_current_sol in frequency.keys():
                            frequency[tuple_current_sol] = SEC_PHASE_DISTURBED_VALUE
                            self.best_sol = current_sol
                            self.min_dist = current_dist_value
                        else:
                            current_dist_value += frequency[
                                tuple_current_sol
                            ]  # penalize by frequency
                            frequency[tuple_current_sol] += SEC_PHASE_DISTURBED_VALUE

                            if current_dist_value < self.min_dist:
                                self.best_sol = current_sol
                                self.min_dist = current_dist_value

            num_iter -= 1

        if print_cost_info:
            print(
                "The cost of solution after Tabu search is: ",
                self.min_dist,
            )
            print("The best solution: ", self.best_sol)

        # Update the mapping result.
        for index in range(len(self.best_sol)):
            ele = self.best_sol[index]
            if ele >= 0:
                vir_qubit_idx = self.vir_qubit_n_tmp_idx.inverse[ele]
                phy_qubit_idx = self.phy_qubit_n_tmp_idx.inverse[index]
                mapping_result[vir_qubit_idx] = phy_qubit_idx

        if is_return_dist:
            return self.min_dist, mapping_result
        else:
            return mapping_result

    def _create_init_sol(self, num_phy_qubits: int, num_vir_qubits: int) -> List[int]:
        """
        Create an initial mapping between virtual qubits and physical qubits.
        """
        vir_qubits_tmp_idx = list(self.vir_qubit_n_tmp_idx.values())
        phy_qubits_tmp_idx = list(self.phy_qubit_n_tmp_idx.values())
        random_vir_qubits_tmp_idx = random.sample(vir_qubits_tmp_idx, num_vir_qubits)
        random_phy_qubits_tmp_idx = random.sample(phy_qubits_tmp_idx, num_vir_qubits)

        current_sol = [-(i + 1) for i in range(num_phy_qubits)]
        for i in range(num_vir_qubits):
            current_sol[random_phy_qubits_tmp_idx[i]] = random_vir_qubits_tmp_idx[i]

        return current_sol

    def _classify_ops(
        self, quantum_ops: List[CircuitInstruction]
    ) -> Tuple[List[CircuitInstruction], List[CircuitInstruction]]:
        """
        Classify quantum operations as local and remote.
        """
        rm_2q_ops = []
        local_2q_ops = []

        for insn in quantum_ops:
            if insn[0].name == "cx" or insn[0].name == "cz":
                qubits = insn[1]
                fst_vir_qubit, sec_vir_qubit = qubits[0], qubits[1]

                if (
                    fst_vir_qubit in self.vir_qubits
                    and sec_vir_qubit not in self.vir_qubits
                ):
                    rm_2q_ops.append(insn)
                elif (
                    fst_vir_qubit not in self.vir_qubits
                    and sec_vir_qubit in self.vir_qubits
                ):
                    rm_2q_ops.append(insn)
                else:
                    local_2q_ops.append(insn)

        return rm_2q_ops, local_2q_ops

    def _create_flow_matrix(self) -> np.ndarray:
        """
        Create the flow matrix of the quantum circuit
        """
        num_vir_qubits = len(self.vir_qubit_n_tmp_idx)
        flow_matrix = np.zeros((num_vir_qubits, num_vir_qubits), dtype=int)

        # The remote op and local op should be distinguished
        # Generate the flow matrix
        for insn in self.local_2q_ops:
            qubits = insn[1]
            fst_tmp_idx, sec_tmp_idx = (
                self.vir_qubit_n_tmp_idx[qubits[0]],
                self.vir_qubit_n_tmp_idx[qubits[1]],
            )
            flow_matrix[fst_tmp_idx][sec_tmp_idx] += 1
            flow_matrix[sec_tmp_idx][fst_tmp_idx] += 1

        return flow_matrix

    def _swap_move(
        self,
        num_qubits: int,
        current_sol: List[int],
        neighbors: np.ndarray,
    ) -> None:
        """
        Generate the neighbors of the current solution.
        """
        neighbor_idx = -1
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i < j:
                    neighbor_idx += 1
                    current_sol[j], current_sol[i] = current_sol[i], current_sol[j]
                    neighbors[neighbor_idx, :-2] = current_sol
                    neighbors[neighbor_idx, -2:] = [current_sol[i], current_sol[j]]
                    current_sol[i], current_sol[j] = current_sol[j], current_sol[i]

    def _not_in_tabu_list(self, solution_tag: Any, tabu_list: List) -> bool:
        """
        Check if the solution is in the tabu list.
        """
        not_found = False
        if not solution_tag.tolist() in tabu_list:
            solution_tag[0], solution_tag[1] = solution_tag[1], solution_tag[0]
            if not solution_tag.tolist() in tabu_list:
                not_found = True
        return not_found

    def _estimate_dist_value(self, solution: List[int]) -> float:
        """
        Calculate the distance value of the current placement.
        """
        dist_value = 0
        num_phy_qubits = len(solution)

        # Calculate the first part of the distance value
        for i in range(num_phy_qubits):
            for j in range(num_phy_qubits):
                if i != j and solution[i] >= 0 and solution[j] >= 0:
                    flow_value = self.flow_matrix[solution[i]][solution[j]]
                    if flow_value > 0:
                        fst_qubit_idx = self.phy_qubit_n_tmp_idx.inverse[i]
                        sec_qubit_idx = self.phy_qubit_n_tmp_idx.inverse[j]
                        dist_value += (
                            self.cost_matrix[fst_qubit_idx][sec_qubit_idx] * flow_value
                        )

        # Calculate the second part of the distance value
        for insn in self.rm_2q_ops:
            fst_qubit = insn[1][0]
            sec_qubit = insn[1][1]

            local_phy_qubit_tmp_idx = None
            nearest_phy_qubit_tmp_idx = None

            if fst_qubit not in self.vir_qubits:
                # If the first qubit is the communication virtual qubit that belongs to another quantum chip

                # Get the nearest physical communication qubit
                tgt_chip_idx = self.assign_res[self.par_res[fst_qubit]]
                source_chip_idx = self.assign_res[self.par_res[sec_qubit]]
                chip_idx_pair = (tgt_chip_idx, source_chip_idx)
                rev_chip_idx_pair = (source_chip_idx, tgt_chip_idx)

                if chip_idx_pair in self.chips_commu_info:
                    _, nearest_commu_qubit = self.chips_commu_info[chip_idx_pair]
                    nearest_phy_qubit_tmp_idx = self.phy_qubit_n_tmp_idx[
                        nearest_commu_qubit
                    ]

                if rev_chip_idx_pair in self.chips_commu_info:
                    nearest_commu_qubit, _ = self.chips_commu_info[rev_chip_idx_pair]
                    nearest_phy_qubit_tmp_idx = self.phy_qubit_n_tmp_idx[
                        nearest_commu_qubit
                    ]

                local_vir_qubit_tmp_idx = self.vir_qubit_n_tmp_idx[sec_qubit]
                local_phy_qubit_tmp_idx = solution.index(local_vir_qubit_tmp_idx)
            elif sec_qubit not in self.vir_qubits:
                # If the second qubit is the communication virtual qubit that belongs to another quantum chip

                # Get the nearest physical communication qubit
                tgt_chip_idx = self.assign_res[self.par_res[sec_qubit]]
                source_chip_idx = self.assign_res[self.par_res[fst_qubit]]
                chip_idx_pair = (tgt_chip_idx, source_chip_idx)
                rev_chip_idx_pair = (source_chip_idx, tgt_chip_idx)

                if chip_idx_pair in self.chips_commu_info:
                    _, nearest_commu_qubit = self.chips_commu_info[chip_idx_pair]
                    nearest_phy_qubit_tmp_idx = self.phy_qubit_n_tmp_idx[
                        nearest_commu_qubit
                    ]

                if rev_chip_idx_pair in self.chips_commu_info:
                    nearest_commu_qubit, _ = self.chips_commu_info[rev_chip_idx_pair]
                    nearest_phy_qubit_tmp_idx = self.phy_qubit_n_tmp_idx[
                        nearest_commu_qubit
                    ]

                local_vir_qubit_tmp_idx = self.vir_qubit_n_tmp_idx[fst_qubit]
                local_phy_qubit_tmp_idx = solution.index(local_vir_qubit_tmp_idx)

            # The basic distance value.
            tmp_dist = self.cost_matrix[
                self.phy_qubit_n_tmp_idx.inverse[local_phy_qubit_tmp_idx]
            ][self.phy_qubit_n_tmp_idx.inverse[nearest_phy_qubit_tmp_idx]]
            dist_value += tmp_dist

        return dist_value


def create_sub_qcs_info(
    par_res: Dict[Qubit, int], quantum_circuit: QuantumCircuit
) -> Tuple[
    Dict[int, List[int]],
    List[int],
    Dict[int, List[CircuitInstruction]],
    List[CircuitInstruction],
]:
    """
    According to the quantum circuit partition result, the corresponding quantum sub-circuit information is generated.
    """
    qubit_blocks = {}
    commu_qubits = []
    operation_blocks = {}
    remote_ops = []

    # Generate the information of qubit blocks.
    for qubit, block_id in par_res.items():
        if block_id in qubit_blocks:
            qubit_blocks[block_id].append(qubit)
        else:
            qubit_blocks[block_id] = []
            qubit_blocks[block_id].append(qubit)

    # Initialize the operation blocks.
    blocks_id = list(qubit_blocks.keys())
    for block_id in blocks_id:
        operation_blocks[block_id] = []

    # Update the information of quantum operation blocks.
    for insn in quantum_circuit:
        qubits = insn.qubits

        insert_blocks_id = []  # The blocks that need to insert the operation.
        for qubit in qubits:
            block_id = par_res[qubit]
            if block_id not in insert_blocks_id:
                insert_blocks_id.append(block_id)
        for id in insert_blocks_id:
            operation_blocks[id].append(insn)

        #! Currently, only consider the cx and cz gate.
        if len(insert_blocks_id) > 1:
            if insn[0].name == "cx" or insn[0].name == "cz":
                if qubits[0] not in commu_qubits:
                    commu_qubits.append(qubits[0])
                if qubits[1] not in commu_qubits:
                    commu_qubits.append(qubits[1])

                # Collect the remote two-qubit quantum operations.
                remote_ops.append(insn)

    return (qubit_blocks, commu_qubits, operation_blocks, remote_ops)


def qubit_group_assign_job(args):
    qg_assign_obj = GlobalAllocation()
    epr_value, allocation_res = qg_assign_obj.search_opt_qubit_group_alloc(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
    )

    return args[11], epr_value, allocation_res


def qg_assign_job_global_calculate(args):
    qg_assign_obj = GlobalAllocation()
    epr_value, allocation_res, mapping_res = qg_assign_obj.global_opt_qubit_group_alloc(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        args[7],
        args[8],
        args[9],
        args[10],
        args[11],
        args[12],
    )

    return args[13], epr_value, allocation_res, mapping_res


def intra_chip_map_job(pool_info):
    local_mapper = IntraProcessMapper()  #! TODO

    local_map_res = local_mapper.run()  #! TODO

    return local_map_res


def single_chip_map_job(pool_info):
    is_return_cost = pool_info[10]

    local_mapper = LocalInitMapper()
    if is_return_cost:
        cost_value, mapping_res = local_mapper.obtain_local_init_mapping(
            pool_info[0],
            pool_info[1],
            pool_info[2],
            pool_info[3],
            pool_info[4],
            pool_info[5],
            pool_info[6],
            pool_info[7],
            pool_info[8],
            pool_info[9],
            pool_info[10],
            pool_info[11],
        )

        return pool_info[12], cost_value, mapping_res
    else:
        mapping_res = local_mapper.obtain_local_init_mapping(
            pool_info[0],
            pool_info[1],
            pool_info[2],
            pool_info[3],
            pool_info[4],
            pool_info[5],
            pool_info[6],
            pool_info[7],
            pool_info[8],
            pool_info[9],
            pool_info[10],
            pool_info[11],
        )

        return pool_info[12], mapping_res
