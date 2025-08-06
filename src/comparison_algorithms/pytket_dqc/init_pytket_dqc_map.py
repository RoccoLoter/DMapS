import time
import networkx as nx
from bidict import bidict
from qiskit import QuantumCircuit
from pytket.extensions.qiskit import qiskit_to_tk
from comparison_algorithms.pytket_dqc.utils.gateset import DQCPass
from comparison_algorithms.pytket_dqc.networks.nisq_network import NISQNetwork
from comparison_algorithms.pytket_dqc.distributors.partitioning_heterogeneous import (
    PartitioningAnnealing,
    PartitioningHeterogeneous,
    PartitioningHeterogeneousEmbedding
)

from comparison_algorithms.pytket_dqc.distributors.distributor import Distribution
from comparison_algorithms.pytket_dqc.circuits.hypergraph_circuit import HypergraphCircuit
from comparison_algorithms.pytket_dqc.placement.placement import Placement


from frontend.chips_info_reader import ChipsNet


def tket_dqc_map(quantum_circuit: QuantumCircuit, qcn: ChipsNet, method: str = "anneal", tketdqc_rout: str = None):
    chip_topology_graph = qcn.obtain_chip_network()
    vqubits = quantum_circuit.qubits
    ep_pqubits_idx = qcn.get_each_chip_qubits_idx()
    ep_capacity = qcn.get_each_chip_capacity()
    pqubits_degree = qcn.obtain_qubit_degree()
    vqubits_degree = _obtain_degree_info(quantum_circuit)

    start_time = time.time()

    # Obtain the useful quantum chips
    useful_chips = _obtain_useful_chips(
        num_vqubits=len(vqubits),
        ep_capacity=ep_capacity,
        chip_topology_graph=chip_topology_graph,
    )

    # Get the related quantum chip network
    server_coupling = []
    server_qubits = {chip: ep_pqubits_idx[chip] for chip in useful_chips}
    
    edges = chip_topology_graph.edges
    for edge in edges:
        if edge[0] not in useful_chips or edge[1] not in useful_chips:
            continue

        if [edge[0], edge[1]] not in server_coupling and [
            edge[1],
            edge[0],
        ] not in server_coupling:
            server_coupling.append([edge[0], edge[1]])

    ep_commu_qubits_idx = qcn.get_each_chip_commu_qubits_idx()
    ep_comm_qubit_cap = {chip_idx : len(qubits) for chip_idx, qubits in ep_commu_qubits_idx.items() if chip_idx in useful_chips}
    network = NISQNetwork(server_coupling=server_coupling, server_qubits=server_qubits, server_ebit_mem=ep_comm_qubit_cap)

    # Convert the quantum circuit to pytket Circuit
    vq_idx_list = [q.index for q in vqubits]
    tk_circ = qiskit_to_tk(quantum_circuit)
    DQCPass().apply(tk_circ)
    

    tmp_placement = None
    if method == "anneal":
        circ_parter = PartitioningAnnealing()
        tket_dqc_res = circ_parter.distribute(tk_circ, network)
        tmp_placement = tket_dqc_res.placement.to_dict()
    elif method == "kahypar":
        #TODO: The Kahypar method is not implemented yet
        circ_parter = PartitioningHeterogeneous()
        tket_dqc_res = circ_parter.distribute(tk_circ, network)
        tmp_placement = tket_dqc_res.placement.to_dict()
    
    epr_cost = 0
    if tketdqc_rout is not None:
        if tketdqc_rout == "distribution":
            new_placement = Placement(tmp_placement)
            distribution = Distribution(HypergraphCircuit(tk_circ), new_placement, network)
            epr_cost = distribution.cost()
    

    final_placement = {
        vq: chip_idx for vq, chip_idx in tmp_placement.items() if vq in vq_idx_list
    }

    # Parse the qubit grouping result
    chips_idx = list(server_qubits.keys())
    qubit_group_res = {chip: [] for chip in chips_idx}
    for vq, idx in final_placement.items():
        qubit_group_res[idx].append(vq)

    # Generate the original qubit mapping
    tmp_init_map = bidict()
    sorted_vqubits_degree = sorted(vqubits_degree.items(), key=lambda x: x[1], reverse=True)
    sorted_pqubits_degree = sorted(pqubits_degree.items(), key=lambda x: x[1], reverse=True)
    for chip_idx, group in qubit_group_res.items():
        pq_list = server_qubits[chip_idx]
        block_sorted_vqubits = [vq for vq, _ in sorted_vqubits_degree if vq.index in group]
        block_sorted_pqubits = [pq for pq, _ in sorted_pqubits_degree if pq in pq_list]

        # tmp_map = dict(zip(group, pq_list))
        tmp_map = dict(zip(block_sorted_vqubits, block_sorted_pqubits))

        for vq, pq in tmp_map.items():
            tmp_init_map[vq] = pq

    # Generate the initial qubit mapping
    # init_map = {vq: tmp_init_map[vq.index] for vq in quantum_circuit.qubits}
    init_map = tmp_init_map

    end_time = time.time()
    time_cost = end_time - start_time

    if tketdqc_rout is not None:
        return time_cost, qubit_group_res, init_map, epr_cost
    else:   
        return time_cost, qubit_group_res, init_map

def _obtain_degree_info(quantum_circuit: QuantumCircuit):
    vqubits_degree = {qubit : 0 for qubit in quantum_circuit.qubits}
    for insn in quantum_circuit:
        if insn.operation.name == "cx" or insn.operation.name == "cz":
            fst_qubit, snd_qubit = insn.qubits[0], insn.qubits[1]
            vqubits_degree[fst_qubit] += 1
            vqubits_degree[snd_qubit] += 1
    return vqubits_degree


def _obtain_useful_chips(
    num_vqubits: int,
    ep_capacity: dict,
    chip_topology_graph: nx.MultiGraph,
):
    sorted_ep_capacity = sorted(ep_capacity.items(), key=lambda x: x[1], reverse=True)
    add_sum_pqubits = sorted_ep_capacity[0][1]
    used_nodes = [sorted_ep_capacity[0][0]]

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

    return used_nodes
