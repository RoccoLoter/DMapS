import time
import math
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from bidict import bidict
from typing import List, Dict
from qiskit import QuantumCircuit
from itertools import combinations

C_EPR = 7
C_BSM = 3
C_RCN = 5


class MHSAMapper:
    """Multistage hybrid simulated annealing algorithm(MHSA) by combining the local search algorithm and a simulated annealing algorithm."""

    def __init__(
        self, alpha: float = 0.9, n_stuck: int = 5, n_stage: int = 50, n_eqlb: int = 5
    ) -> None:
        # The information of the quantum chip network
        self.chips_capacity = {}
        self.total_capacity = 0
        self.chip_dist_matrix = None

        # The information of the quantum circuit
        self.ops_degree = None
        self.qubits_n_idx = bidict()

        # The parameters of MSHA algorithm
        self.init_temperature = 100.0
        self.stop_temperature = 1e-3
        self.cur_temperature = 0.0
        self.alpha = alpha
        self.n_stage = n_stage  # Inner loop iteration number
        self.n_stuck = n_stuck
        self.n_eqlb = n_eqlb

        self.index_n_chip = {}
        self.all_index_combinations = []

        self.run_time = 0

    def run(
        self,
        quantum_circuit: QuantumCircuit,
        each_chip_qubits: Dict[int, List[int]],
        pqubits_degree: Dict[int, int],
        chip_dist_matrix: np.ndarray,
        qcn=None,
    ):
        qubit_alloc_res = {}

        start_time = time.time()

        # Obtain the capacity information of each quantum chip.
        chips_capacity = {chip_idx: len(qubit_list) for chip_idx, qubit_list in each_chip_qubits.items()}

        # Obtain the useful quantum chips
        chip_topology_graph = qcn.obtain_chip_network()
        useful_chips, _ = self._obtain_useful_chips(
            num_vqubits=len(quantum_circuit.qubits), 
            ep_capacity=chips_capacity, 
            chip_topology_graph=chip_topology_graph
        )
        # ordered_useful_chips = self._sort_nodes_by_bfs(useful_chips_connect_graph, useful_chips)
        useful_chips_capacity = {chip: chips_capacity[chip] for chip in useful_chips}

        # # TODO: Test
        # useful_chips_capacity = {0 : 12, 4 : 12, 2 : 11}

        opt_partition = self._run_partition(
            quantum_circuit, useful_chips_capacity, chip_dist_matrix
        )
        qubit_alloc_res = self._generate_local_allocation(
            opt_partition, each_chip_qubits, quantum_circuit, pqubits_degree
        )

        end_time = time.time()
        self.run_time = end_time - start_time

        return qubit_alloc_res

    def _run_partition(
        self,
        quantum_circuit: QuantumCircuit,
        chips_capacity: Dict[int, int],
        chip_dist_matrix: np.ndarray,
    ):
        """Perform the partitioning of the quantum circuit."""
        best_sol = []
        best_energy = 0.0

        # The information about the quantum chip network
        self.chips_capacity = chips_capacity
        self.chip_dist_matrix = chip_dist_matrix
        for _, capacity in self.chips_capacity.items():
            self.total_capacity += capacity

        # Get the related information of the quantum circuit
        self.qubits_n_idx, self.ops_degree = self._circuit_info(quantum_circuit)

        qubit_idx_list = list(self.qubits_n_idx.keys())
        # Generate a initial partitioning result
        init_par = [-i for i in range(1, self.total_capacity + 1)]

        par_list_idx = 0
        for ele in qubit_idx_list:
            init_par[par_list_idx] = ele
            par_list_idx += 1

        # Check if the length of the list is equal to the sum of the chip capacities
        assert len(init_par) == self.total_capacity
        # The index range information of each quantum chip in the partitoning result list
        self.index_n_chip, each_chip_index_scale, self.all_index_combinations = (
            self._chips_rela_info()
        )

        """The main progress of the MHSA algorithm"""
        cur_sol = init_par
        best_sol = cur_sol
        best_energy = self._objective_function(best_sol)
        cur_energy = best_energy
        self.cur_temperature = self.init_temperature

        tmp_n_eqlb = 0  # Equalibrium test
        tmp_n_stuck = 0  # Termination test
        terminate_flag = False
        while not terminate_flag and self.cur_temperature > self.stop_temperature:
            for _ in range(self.n_stage):
                new_energy, opt_new_sol = self._opt_local_search(cur_sol)

                # Determine whether to accept the new solution
                if self._accept_new_solution(cur_energy, new_energy):
                    cur_sol = opt_new_sol
                    cur_energy = new_energy

            if cur_energy < best_energy:
                best_sol = cur_sol
                best_energy = cur_energy
            else:
                tmp_n_stuck += 1

            # Update the status of termination flag
            if tmp_n_stuck == self.n_stuck:
                terminate_flag = True
                continue

            # Equalibrium test
            tmp_n_eqlb += 1
            if tmp_n_eqlb == self.n_eqlb:
                self.cur_temperature *= self.alpha  # Update the temperature
                tmp_n_eqlb = 0

        final_partition = self._obtain_partition_info(best_sol, each_chip_index_scale)

        return final_partition

    def _generate_local_allocation(
        self, partition: Dict[int, List[int]], each_chip_qubits: Dict[int, List[int]], quantum_circuit: QuantumCircuit, pqubits_degree: Dict[int, int]
    ):
        """Obtain the mapping results of qubits within each quantum chip based on the partitioning results."""
        total_qubit_alloc = bidict()

        vqubits_degree = self._obtain_degree_info(quantum_circuit)
        sorted_vqubits_degree = sorted(vqubits_degree.items(), key=lambda x: x[1], reverse=True)
        sorted_pqubits_degree = sorted(pqubits_degree.items(), key=lambda x: x[1], reverse=True)

        for chip_idx, par_block in partition.items():
            phy_qubit_list = each_chip_qubits[chip_idx]

            # Check the capacity of the quantum chip
            assert len(phy_qubit_list) >= len(
                par_block
            )  

            block_sorted_vqubits = [vq for vq, _ in sorted_vqubits_degree if vq in par_block]
            block_sorted_pqubits = [pq for pq, _ in sorted_pqubits_degree if pq in phy_qubit_list]

            # tmp_mapping = dict(zip(par_block, phy_qubit_list))
            tmp_mapping = dict(zip(block_sorted_vqubits, block_sorted_pqubits))
            for vir_qubit, phy_qubit in tmp_mapping.items():
                total_qubit_alloc[vir_qubit] = phy_qubit

        return total_qubit_alloc

    def _obtain_partition_info(
        self, solution: List[int], each_chip_index_scale: Dict[int, List[int]]
    ):
        """Generate the formal partitioning result."""
        partition = {}
        for chip_idx, idx_scale in each_chip_index_scale.items():
            partition[chip_idx] = []
            for idx in idx_scale:
                ele = solution[idx]
                if ele >= 0:
                    partition[chip_idx].append(self.qubits_n_idx[ele])

        return partition

    def _circuit_info(self, quantum_circuit: QuantumCircuit):
        """Get the related information of the quantum circuit."""
        ops_degree = {}

        qubits_n_idx = bidict(enumerate(quantum_circuit.qubits))

        for insn in quantum_circuit:
            if insn.operation.name == "cx" or insn.operation.name == "cz":
                fst_qubit, snd_qubit = insn.qubits[0], insn.qubits[1]
                fst_qubit_idx, snd_qubit_idx = (
                    qubits_n_idx.inverse[fst_qubit],
                    qubits_n_idx.inverse[snd_qubit],
                )

                qubits_pair = (fst_qubit_idx, snd_qubit_idx)
                rev_qubits_pair = (snd_qubit_idx, fst_qubit_idx)
                if qubits_pair not in ops_degree and rev_qubits_pair not in ops_degree:
                    ops_degree[qubits_pair] = 1
                elif qubits_pair in ops_degree and rev_qubits_pair not in ops_degree:
                    ops_degree[qubits_pair] += 1
                elif qubits_pair not in ops_degree and rev_qubits_pair in ops_degree:
                    ops_degree[rev_qubits_pair] += 1

        return qubits_n_idx, ops_degree

    def _obtain_degree_info(self, quantum_circuit: QuantumCircuit):
        vqubits_degree = {qubit : 0 for qubit in quantum_circuit.qubits}

        for insn in quantum_circuit:
            if insn.operation.name == "cx" or insn.operation.name == "cz":
                fst_qubit, snd_qubit = insn.qubits[0], insn.qubits[1]
                vqubits_degree[fst_qubit] += 1
                vqubits_degree[snd_qubit] += 1

        return vqubits_degree

    def _objective_function(self, cur_sol: List[int]):
        """Evaluate the partitioning result."""
        objective_value = 0

        qubit_n_chip = {}
        for idx in range(len(cur_sol)):
            qubit_n_chip[cur_sol[idx]] = self.index_n_chip[idx]

        for op_pair, degree in self.ops_degree.items():
            fst_qubit, snd_qubit = op_pair[0], op_pair[1]
            fst_chip_idx, snd_chip_idx = (
                qubit_n_chip[fst_qubit],
                qubit_n_chip[snd_qubit],
            )

            if fst_chip_idx != snd_chip_idx:
                chip_dist = self.chip_dist_matrix[fst_chip_idx][snd_chip_idx]
                tmp_cost_value = chip_dist * C_EPR + (chip_dist - 1) * C_BSM + C_RCN
                objective_value += tmp_cost_value * degree

        return objective_value

    def _chips_rela_info(self):
        """The index range information of each quantum chip in the partitoning result list."""
        index_n_chip = {}
        each_chip_index_scale = {}
        all_combinations = []

        # Obtain the index range information of each quantum chip
        part_size = 0
        for chip_idx, capacity in self.chips_capacity.items():
            tmp_list = list(range(part_size, part_size + capacity))
            each_chip_index_scale[chip_idx] = tmp_list

            for idx in tmp_list:
                index_n_chip[idx] = chip_idx

            part_size += capacity

        # Get the combinations of two different list index that belong to different quantum chips
        chip_combiantions = combinations(self.chips_capacity.keys(), 2)
        for fst_chip, snd_chip in chip_combiantions:
            fst_idx_scale = each_chip_index_scale[fst_chip]
            snd_idx_scale = each_chip_index_scale[snd_chip]

            # Get the combinations of two different list indices
            for fst_idx in fst_idx_scale:
                for snd_idx in snd_idx_scale:
                    all_combinations.append((fst_idx, snd_idx))

        return index_n_chip, each_chip_index_scale, all_combinations

    def _opt_local_search(self, cur_sol: List[int]) -> List[int]:
        """Find the best neighbor solution of the current solution."""
        best_neighbor = None
        minal_energy = float("inf")

        for fst_idx, snd_idx in self.all_index_combinations:
            if cur_sol[fst_idx] < 0 and cur_sol[snd_idx] < 0:
                continue

            cur_sol[fst_idx], cur_sol[snd_idx] = cur_sol[snd_idx], cur_sol[fst_idx]
            tmp_sol = deepcopy(cur_sol)
            cur_sol[fst_idx], cur_sol[snd_idx] = cur_sol[snd_idx], cur_sol[fst_idx]
            tmp_energy = self._objective_function(tmp_sol)
            if tmp_energy < minal_energy:
                minal_energy = tmp_energy
                best_neighbor = deepcopy(tmp_sol)

        return minal_energy, best_neighbor

    def _accept_new_solution(self, cur_energy: float, new_energy: float) -> bool:
        """Determine whether to accept the new solution."""
        if new_energy < cur_energy:
            return True
        else:
            accept_prob = math.exp(-(new_energy - cur_energy) / self.cur_temperature)
            return random.random() < accept_prob


    def _obtain_useful_chips(self, num_vqubits: int, ep_capacity: dict, chip_topology_graph: nx.MultiGraph):
        sorted_ep_capacity = sorted(ep_capacity.items(), key=lambda x: x[1], reverse=True)
        add_sum_pqubits = sorted_ep_capacity[0][1]
        used_nodes = [sorted_ep_capacity[0][0]]
        new_chip_connect_graph = nx.Graph()
        new_chip_connect_graph.add_node(used_nodes[0])

        while add_sum_pqubits < num_vqubits:
            neighbor_nodes = []

            for used_node in used_nodes:
                for neighbor in chip_topology_graph.neighbors(used_node):
                    if neighbor not in used_nodes and neighbor not in neighbor_nodes:
                        neighbor_nodes.append(neighbor)

            sorted_neighbor_nodes = sorted(
                neighbor_nodes, key=lambda x: ep_capacity[x], reverse=True
            )
            selected_node = sorted_neighbor_nodes[0]
            add_sum_pqubits += ep_capacity[selected_node]
            used_nodes.append(selected_node)

            neighbor_selected_node = chip_topology_graph.neighbors(selected_node)
            for neighbor in neighbor_selected_node:
                if neighbor in used_nodes:
                    new_chip_connect_graph.add_edge(selected_node, neighbor)
                    break
        
        random.shuffle(used_nodes)

        return used_nodes, new_chip_connect_graph
    
    def _sort_nodes_by_bfs(self, graph, node_set):
        """Sort the vertices in graph in BFS traversal order"""
        start_node = min(node_set, key=lambda node: graph.degree[node])

        bfs_order = list(nx.bfs_tree(graph, start_node))

        return [node for node in bfs_order if node in node_set] 