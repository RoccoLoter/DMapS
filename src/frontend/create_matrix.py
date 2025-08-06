import numpy as np
import networkx as nx
from typing import Dict
from itertools import combinations
from networkx import all_simple_paths

ADJACENT_CHIP_COMM_COST = 10

def create_cost_matrix(
    qubits_topology: nx.Graph, commu_qubits_info: Dict[int, int], remote_dist=None
):
    """Create the related matrix of qubits topology considered the remote physical connections."""
    remote_connection_dist = 0
    if remote_dist is None:
        remote_connection_dist = ADJACENT_CHIP_COMM_COST
    else:
        remote_connection_dist = remote_dist

    dist_graph = nx.Graph()
    qubits_idx = list(qubits_topology.nodes())
    dist_matrix_consider_comm = np.ones((len(qubits_idx), len(qubits_idx))) * np.inf

    for fst_qubit_idx, snd_qubit_idx in qubits_topology.edges():
        if fst_qubit_idx in commu_qubits_info and snd_qubit_idx in commu_qubits_info:
            if commu_qubits_info[fst_qubit_idx] != commu_qubits_info[snd_qubit_idx]:
                dist_graph.add_edge(
                    fst_qubit_idx, snd_qubit_idx, weight=remote_connection_dist
                )
            else:
                dist_graph.add_edge(fst_qubit_idx, snd_qubit_idx, weight=1)
        else:
            dist_graph.add_edge(fst_qubit_idx, snd_qubit_idx, weight=1)

    # Calculate the cost value of nearest path between two qubit.
    dists = nx.floyd_warshall(dist_graph)
    for node, distances_from_node in dists.items():
        for neighbor, distance in distances_from_node.items():
            dist_matrix_consider_comm[node][neighbor] = distance

    return dist_matrix_consider_comm


def create_dist_matrix(qubits_topology: nx.Graph) -> np.ndarray:
    """Create the related matrix of qubits topology considered the remote physical connections."""
    qubits_idx = list(qubits_topology.nodes())
    dist_matrix = np.ones((len(qubits_idx), len(qubits_idx))) * np.inf

    for qubit_idx in qubits_idx:
        dist_matrix[qubit_idx][qubit_idx] = 0

    for fst_qubit_idx, sec_qubit_idx in qubits_topology.edges():
        dist_matrix[fst_qubit_idx][sec_qubit_idx] = 1
        dist_matrix[sec_qubit_idx][fst_qubit_idx] = 1

    # Calculate the cost value of nearest path between two qubit.
    for u in qubits_idx:
        for v in qubits_idx:
            for w in qubits_idx:
                tmp_value = dist_matrix[u][v] + dist_matrix[v][w]
                if tmp_value < dist_matrix[u][w]:
                    dist_matrix[u][w] = tmp_value
                    dist_matrix[w][u] = tmp_value

    return dist_matrix


def create_chip_dist_matrix(chips_topology: nx.MultiGraph) -> np.ndarray:
    """Create the distance matrix of quantum chip to quantum chip."""
    chips_idx = list(chips_topology.nodes())
    chips_dist_matrix = np.ones((len(chips_idx), len(chips_idx))) * np.inf

    for chip_idx in chips_idx:
        chips_dist_matrix[chip_idx][chip_idx] = 0
    for fst_chip_idx, sec_chip_idx in chips_topology.edges():
        if chips_dist_matrix[fst_chip_idx][sec_chip_idx] == np.inf:
            chips_dist_matrix[fst_chip_idx][sec_chip_idx] = 1
            chips_dist_matrix[sec_chip_idx][fst_chip_idx] = 1

    for u in chips_idx:
        for v in chips_idx:
            for w in chips_idx:
                tmp_dist = chips_dist_matrix[u][v] + chips_dist_matrix[v][w]
                if tmp_dist < chips_dist_matrix[u][w]:
                    chips_dist_matrix[u][w] = tmp_dist
                    chips_dist_matrix[w][u] = tmp_dist

    return chips_dist_matrix


def create_multi_fid_matrix(qubits_topology: nx.Graph) -> np.ndarray:
    """Create the multiple fidelity matrix of qubits topology."""
    qubits_idx = list(qubits_topology.nodes())
    distance_matrix = np.zeros((len(qubits_idx), len(qubits_idx)))

    # Initialize the distance matrix with the fidelity of qubit connection.
    for qubit_idx in qubits_idx:
        distance_matrix[qubit_idx, qubit_idx] = 1.0
    for fst_qubit_idx, sec_qubit_idx, edge_info in qubits_topology.edges(data=True):
        distance_matrix[fst_qubit_idx, sec_qubit_idx] = edge_info["fidelity"]
        distance_matrix[sec_qubit_idx, fst_qubit_idx] = edge_info["fidelity"]

    # Calculate the strongest path between two qubit.
    for u in qubits_idx:
        for v in qubits_idx:
            for w in qubits_idx:
                tmp_value = distance_matrix[u][v] * distance_matrix[v][w]
                if tmp_value > distance_matrix[u][w]:
                    distance_matrix[u][w] = tmp_value
                    distance_matrix[w][u] = tmp_value
    return distance_matrix


def create_robustness_matrix(qubits_topology: nx.Graph) -> np.ndarray:
    """Create the robustness matrix of qubits topology."""
    qubits_idx = list(qubits_topology.nodes())
    robustness_matrix = np.zeros((len(qubits_idx), len(qubits_idx)))

    # Initialize the fidelity matrix with the fidelity of qubit connection.
    for qubit_idx in qubits_idx:
        robustness_matrix[qubit_idx][qubit_idx] = 1.0
    for qubit_1_idx, qubit_2_idx, edge_info in qubits_topology.edges(data=True):
        robustness_matrix[qubit_1_idx][qubit_2_idx] = edge_info["fidelity"]
        robustness_matrix[qubit_2_idx][qubit_1_idx] = edge_info["fidelity"]

    # Calculate the fidelity result of strongest path between two physical qubit.
    phy_qubit_pairs = combinations(qubits_idx, 2)
    for qubit_pair in phy_qubit_pairs:
        fst_idx, snd_idx = qubit_pair

        # Get the information of all simple paths between two physical qubit.
        simple_paths = all_simple_paths(qubits_topology, fst_idx, snd_idx)
        max_rb_value = 0
        for path in simple_paths:
            coupling_fid_list = []

            i = 0
            while i < len(path) - 1:
                coupling_fid_list.append(robustness_matrix[path[i]][path[i + 1]])
                i += 1
            coupling_fid_list.sort(reverse=True)

            tmp_rb_value = 1.0
            j = 0
            while j < len(coupling_fid_list):
                if j != len(coupling_fid_list) - 1:
                    tmp_rb_value *= pow(coupling_fid_list[j], 3)
                else:
                    tmp_rb_value *= coupling_fid_list[j]
                j += 1

            if tmp_rb_value > max_rb_value:
                max_rb_value = tmp_rb_value

        robustness_matrix[fst_idx][snd_idx] = max_rb_value
        robustness_matrix[snd_idx][fst_idx] = max_rb_value

    return robustness_matrix
