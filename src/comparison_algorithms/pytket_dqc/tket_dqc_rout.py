import math
import logging
import retworkx
import numpy as np
from networkx import Graph
from copy import copy, deepcopy
from collections import defaultdict

from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_circuit, circuit_to_dag


logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = (
    10  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)
EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.

DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class tketdqc_local_rout(TransformationPass):
    def __init__(
        self,
        coupling_map,
        chips_net,
        heuristic="lookahead",
        seed=None,
        fake_run=False,
        exe_mode="sabre",
        input_layout=None,
        qubits_topology=None,
        input_matrix_tolist=None,
        lookahead_ability=20,
    ):
        super().__init__()

        # Mending
        self.exe_mode = exe_mode
        self.qubits_topology = qubits_topology
        self.input_layout = input_layout

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.required_predecessors = None
        self.qubits_decay = None
        self._bit_indices = None

        if input_matrix_tolist is not None:
            self.dist_matrix = np.array(input_matrix_tolist)
        else:
            self.dist_matrix = None

        self.extended_set_size = 20
        self.lookahead_ability = lookahead_ability

        self.qubits_name_n_idx = chips_net.qubits_n_index
        # Get the physical qubits index and its related quantum chip index.
        self.phy_qubits_info = {}
        for chip in chips_net.chips:
            for qubit_idx in chip.qubits_index:
                self.phy_qubits_info[qubit_idx] = chip.index

        self.comm_qubit_n_chip = {}
        self.each_chip_commu_qubits = chips_net.get_each_chip_commu_qubits_idx()
        for chip_idx, phy_commu_qubits in self.each_chip_commu_qubits.items():
            for qubit in phy_commu_qubits:
                self.comm_qubit_n_chip[qubit] = chip_idx

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.
        ops_since_progress = []
        extended_set = None

        # Normally this isn't necessary, but here we want to log some objects that have some
        # non-trivial cost to create.
        do_expensive_logging = logger.isEnabledFor(logging.DEBUG)

        # TODO: Mending
        if self.exe_mode == "fha":
            if self.qubits_topology is not None:
                self.dist_matrix = create_distance_matrix_consider_fidelity(
                    self.qubits_topology
                )
            else:
                raise TranspilerError("The qubits topology must be given.")
        elif self.exe_mode == "sabre_fid":
            if self.qubits_topology is not None:
                self.dist_matrix = create_distance_matrix_consider_fidelity(
                    self.qubits_topology
                )
            else:
                raise TranspilerError(
                    "If sabre algorithm consider physical coupling fidelity, the qubits topology must be given."
                )
        elif self.exe_mode == "sabre" or self.exe_mode == "sabre_v2":
            if self.dist_matrix is None:
                if self.coupling_map is not None:
                    self.dist_matrix = self.coupling_map.distance_matrix

        rng = np.random.default_rng(self.seed)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = None
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        # Get the canonical register.
        canonical_register = None
        dag_registers = dag.qregs
        if len(dag_registers) == 1:
            for _, reg in dag_registers.items():
                canonical_register = reg
        else:
            raise TranspilerError("To be realized.")

        current_layout = None
        # Mending
        if self.exe_mode == "fha":
            # Update the current layout.
            if self.input_layout is not None:
                current_layout = self.input_layout
            else:
                current_layout = Layout.generate_trivial_layout(canonical_register)

        elif (
            self.exe_mode == "sabre"
            or self.exe_mode == "sabre_fid"
            or self.exe_mode == "sabre_v2"
        ):
            if self.input_layout is not None:
                current_layout = self.input_layout
            else:
                # TODO: Mending this code
                # canonical_register = dag.qregs["q"]
                current_layout = Layout.generate_trivial_layout(canonical_register)

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        self.qubits_decay = dict.fromkeys(dag.qubits, 1)

        # Start algorithm from the front layer and iterate until all gates done.
        self.required_predecessors = self._build_required_predecessors(dag)
        num_search_steps = 0
        front_layer = dag.front_layer()

        while front_layer:
            execute_gate_list = []

            # Remove as many immediately applicable gates as possible
            new_front_layer = []
            for node in front_layer:
                if len(node.qargs) == 2:
                    v0, v1 = node.qargs
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self.coupling_map.graph.has_edge(
                        current_layout._v2p[v0], current_layout._v2p[v1]
                    ):
                        execute_gate_list.append(node)
                    else:
                        new_front_layer.append(node)
                else:  # Single-qubit gates as well as barriers are free
                    execute_gate_list.append(node)
            front_layer = new_front_layer

            if (
                not execute_gate_list
                and len(ops_since_progress) > max_iterations_without_progress
            ):
                # Backtrack to the last time we made progress, then greedily insert swaps to route
                # the gate with the smallest distance between its arguments.  This is a release
                # valve for the algorithm to avoid infinite loops only, and should generally not
                # come into play for most circuits.
                self._undo_operations(ops_since_progress, mapped_dag, current_layout)
                if self.exe_mode == "sabre_fid":
                    self._add_greedy_swaps_consider_fid(
                        front_layer, mapped_dag, current_layout, canonical_register
                    )
                if (
                    self.exe_mode == "fha"
                    or self.exe_mode == "sabre"
                    or self.exe_mode == "sabre_v2"
                ):
                    self._add_greedy_swaps(
                        front_layer, mapped_dag, current_layout, canonical_register
                    )
                continue

            if execute_gate_list:
                for node in execute_gate_list:
                    self._apply_gate(
                        mapped_dag, node, current_layout, canonical_register
                    )
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

                    if node.qargs:
                        self._reset_qubits_decay()

                # Diagnostics
                if do_expensive_logging:
                    logger.debug(
                        "free! %s",
                        [
                            (n.name if isinstance(n, DAGOpNode) else None, n.qargs)
                            for n in execute_gate_list
                        ],
                    )
                    logger.debug(
                        "front_layer: %s",
                        [
                            (n.name if isinstance(n, DAGOpNode) else None, n.qargs)
                            for n in front_layer
                        ],
                    )

                ops_since_progress = []
                extended_set = None
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            if extended_set is None:
                extended_set = self._obtain_extended_set(dag, front_layer)
            swap_scores = {}
            best_score = 0.0
            for swap_qubits in self._obtain_swaps(front_layer, current_layout):
                trial_layout = current_layout.copy()
                trial_layout.swap(*swap_qubits)

                # TODO: Mending
                if self.exe_mode == "sabre_fid":
                    score = self._score_heuristic_consider_fid(
                        self.heuristic,
                        front_layer,
                        extended_set,
                        trial_layout,
                        swap_qubits,
                    )
                    swap_scores[swap_qubits] = score
                    best_score = max(swap_scores.values())
                elif self.exe_mode == "fha":
                    score = self._fha_score_heuristic(
                        front_layer, extended_set, trial_layout, swap_qubits
                    )
                    swap_scores[swap_qubits] = score
                    best_score = max(swap_scores.values())
                elif self.exe_mode == "sabre":
                    score = self._score_heuristic(
                        self.heuristic,
                        front_layer,
                        extended_set,
                        trial_layout,
                        swap_qubits,
                    )
                    swap_scores[swap_qubits] = score
                    best_score = min(swap_scores.values())
                elif self.exe_mode == "sabre_v2":
                    score = self._score_heuristic_v2(
                        self.heuristic,
                        front_layer,
                        extended_set,
                        trial_layout,
                        swap_qubits,
                    )
                    swap_scores[swap_qubits] = score
                    best_score = min(swap_scores.values())
            best_swaps = [k for k, v in swap_scores.items() if v == best_score]
            best_swaps.sort(
                key=lambda x: (self._bit_indices[x[0]], self._bit_indices[x[1]])
            )
            best_swap = rng.choice(best_swaps)

            # # Test: Output swap information
            # tmp_mapping = current_layout._v2p
            # tmp_data = {v.index: p for v, p in tmp_mapping.items()}
            # current_layout_mapping = conv_mapping_res(tmp_data, self.qubits_name_n_idx)
            # is_remote_swap = self._is_remote_swap(best_swap, current_layout)
            # print(
            #     (
            #         is_remote_swap,
            #         current_layout_mapping[best_swap[0].index],
            #         current_layout_mapping[best_swap[1].index],
            #     )
            # )

            swap_node = self._apply_gate(
                mapped_dag,
                DAGOpNode(op=SwapGate(), qargs=best_swap),
                current_layout,
                canonical_register,
            )
            current_layout.swap(*best_swap)
            ops_since_progress.append(swap_node)

            num_search_steps += 1
            if num_search_steps % DECAY_RESET_INTERVAL == 0:
                self._reset_qubits_decay()
            else:
                self.qubits_decay[best_swap[0]] += DECAY_RATE
                self.qubits_decay[best_swap[1]] += DECAY_RATE

            # Diagnostics
            if do_expensive_logging:
                logger.debug("SWAP Selection...")
                logger.debug(
                    "extended_set: %s", [(n.name, n.qargs) for n in extended_set]
                )
                logger.debug("swap scores: %s", swap_scores)
                logger.debug("best swap: %s", best_swap)
                logger.debug("qubits decay: %s", self.qubits_decay)

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
            # return mapped_dag, current_layout
        return current_layout, dag

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)
        if self.fake_run:
            return new_node
        return mapped_dag.apply_operation_back(
            new_node.op, new_node.qargs, new_node.cargs
        )

    def _reset_qubits_decay(self):
        """Reset all qubit decay factors to 1 upon request (to forget about
        past penalizations).
        """
        self.qubits_decay = {k: 1 for k in self.qubits_decay.keys()}

    def _build_required_predecessors(self, dag):
        out = defaultdict(int)
        # We don't need to count in- or out-wires: outs can never be predecessors, and all input
        # wires are automatically satisfied at the start.
        for node in dag.op_nodes():
            for successor in self._successors(node, dag):
                out[successor] += 1
        return out

    def _successors(self, node, dag):
        """Return an iterable of the successors along each wire from the given node.

        This yields the same successor multiple times if there are parallel wires (e.g. two adjacent
        operations that have one clbit and qubit in common), which is important in the swapping
        algorithm for detecting if each wire has been accounted for."""
        for _, successor, _ in dag.edges(node):
            if isinstance(successor, DAGOpNode):
                yield successor

    def _is_resolved(self, node):
        """Return True if all of a node's predecessors in dag are applied."""
        return self.required_predecessors[node] == 0

    def _obtain_extended_set(self, dag, front_layer):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit.
        """
        # self.extended_set_size = EXTENDED_SET_SIZE
        self.extended_set_size = self.lookahead_ability * len(front_layer)

        extended_set = []
        decremented = []
        tmp_front_layer = front_layer
        done = False
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    decremented.append(successor)
                    self.required_predecessors[successor] -= 1
                    if self._is_resolved(successor):
                        new_tmp_front_layer.append(successor)
                        if len(successor.qargs) == 2:
                            extended_set.append(successor)
                if len(extended_set) >= self.extended_set_size:
                    done = True
                    break
            tmp_front_layer = new_tmp_front_layer
        for node in decremented:
            self.required_predecessors[node] += 1
        return extended_set

    def _obtain_swaps(self, front_layer, current_layout):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()
        for node in front_layer:
            for virtual in node.qargs:
                physical = current_layout[virtual]
                for neighbor in self.coupling_map.neighbors(physical):
                    if neighbor in self.comm_qubit_n_chip and physical in self.comm_qubit_n_chip:
                        if self.comm_qubit_n_chip[neighbor] != self.comm_qubit_n_chip[physical]:
                            continue
                    virtual_neighbor = current_layout[neighbor]
                    swap = sorted(
                        [virtual, virtual_neighbor], key=lambda q: self._bit_indices[q]
                    )
                    candidate_swaps.add(tuple(swap))
        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        target_node = min(
            front_layer,
            key=lambda node: self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ],
        )

        for pair in _shortest_swap_path(
            tuple(target_node.qargs), self.coupling_map, layout
        ):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _add_greedy_swaps_consider_fid(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        target_node = max(
            front_layer,
            key=lambda node: self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ],
        )

        for pair in _shortest_swap_path(
            tuple(target_node.qargs), self.coupling_map, layout
        ):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _compute_cost(self, layer, layout):
        cost = 0
        layout_map = layout._v2p
        for node in layer:
            cost += self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ]
        return cost

    # TODO: Mending
    def _compute_cost_consider_fid(self, layer, layout):
        cost = 1.0
        layout_map = layout._v2p
        for node in layer:
            cost *= self.dist_matrix[
                layout_map[node.qargs[0]], layout_map[node.qargs[1]]
            ]
        return cost

    def _score_heuristic(
        self, heuristic, front_layer, extended_set, layout, swap_qubits=None
    ):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost /= len(front_layer)
        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout) / len(extended_set)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(
                    self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]
                )
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    def _score_heuristic_v2(
        self, heuristic, front_layer, extended_set, layout, swap_qubits=None
    ):
        """Return a heuristic score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a cost
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        first_cost = self._compute_cost(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        second_cost = 0
        if extended_set:
            second_cost = self._compute_cost(extended_set, layout)
        total_cost = first_cost + EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(
                    self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]
                )
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    # TODO: Mending
    def _score_heuristic_consider_fid(
        self, heuristic, front_layer, extended_set, layout, swap_qubits=None
    ):
        first_cost = self._compute_cost_consider_fid(front_layer, layout)
        if heuristic == "basic":
            return first_cost

        first_cost = math.pow(first_cost, 1 / len(front_layer))
        second_cost = 1.0
        if extended_set:
            tmp_second_cost = self._compute_cost_consider_fid(extended_set, layout)
            second_cost = math.pow(tmp_second_cost, 1 / len(extended_set))
        total_cost = first_cost * EXTENDED_SET_WEIGHT * second_cost
        if heuristic == "lookahead":
            return total_cost

        if heuristic == "decay":
            return (
                max(
                    self.qubits_decay[swap_qubits[0]], self.qubits_decay[swap_qubits[1]]
                )
                * total_cost
            )

        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    def _fha_score_heuristic(self, front_layer, extended_set, layout, swap_qubits=None):
        total_cost = 1.0
        tmp_cost = self.dist_matrix[
            layout._v2p[swap_qubits[0]], layout._v2p[swap_qubits[1]]
        ]

        first_cost = tmp_cost * self._compute_cost_consider_fid(front_layer, layout)
        second_cost = 1.0
        if extended_set:
            second_cost = self._compute_cost_consider_fid(extended_set, layout)

        total_cost = first_cost * pow(second_cost, EXTENDED_SET_WEIGHT)

        return total_cost

    def _undo_operations(self, operations, dag, layout):
        """Mutate ``dag`` and ``layout`` by undoing the swap gates listed in ``operations``."""
        if dag is None:
            for operation in reversed(operations):
                layout.swap(*operation.qargs)
        else:
            for operation in reversed(operations):
                dag.remove_op_node(operation)
                p0 = self._bit_indices[operation.qargs[0]]
                p1 = self._bit_indices[operation.qargs[1]]
                layout.swap(p0, p1)

    def _is_remote_swap(self, swap_qubits, current_layout):
        """Judge whether the inserted swap operation is remote."""
        is_remote = False

        current_mapping = current_layout._v2p

        fst_phy_qubit = current_mapping[swap_qubits[0]]
        sec_phy_qubit = current_mapping[swap_qubits[1]]
        fst_chip_idx = self.phy_qubits_info[fst_phy_qubit]
        sec_chip_idx = self.phy_qubits_info[sec_phy_qubit]

        if fst_chip_idx != sec_chip_idx:
            is_remote = True
        else:
            is_remote = False

        return is_remote


def conv_mapping_res(ori_mapping_res, qubits_name_n_idx):
    """Convert the physical qubits in the qubits mapping result."""
    new_mapping_res = {}
    for k, v in ori_mapping_res.items():
        qubit_name = qubits_name_n_idx.inverse[v]
        new_qubit_idx = int(qubit_name[1:])
        new_mapping_res[k] = new_qubit_idx
    return new_mapping_res


def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = [device_qreg[layout._v2p[x]] for x in op_node.qargs]
    return mapped_op_node


# Mending
def _shortest_swap_path(target_qubits, coupling_map, layout):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    v_start, v_goal = target_qubits
    start, goal = layout._v2p[v_start], layout._v2p[v_goal]
    # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
    path = list(
        retworkx.dijkstra_shortest_paths(coupling_map.graph, start, target=goal)[goal]
    )
    # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
    # parallelise and reduce depth.
    split = len(path) // 2
    forwards, backwards = path[1:split], reversed(path[split:-1])
    for swap in forwards:
        yield v_start, layout._p2v[swap]
    for swap in backwards:
        yield v_goal, layout._p2v[swap]


# TODO: It can improve the efficiency of computing.
def create_distance_matrix_consider_fidelity(qubits_topology: Graph) -> np.matrix:
    """Create the distance matrix of qubits topology."""
    qubits_idx = list(qubits_topology.nodes())
    qubits_num = len(qubits_idx)
    distance_matrix = np.zeros((qubits_num, qubits_num))

    # Initialize the distance matrix with the fidelity of qubit connection.
    for qubit_idx in qubits_idx:
        distance_matrix[qubit_idx][qubit_idx] = 1.0
    for qubit_1_idx, qubit_2_idx, edge_info in qubits_topology.edges(data=True):
        distance_matrix[qubit_1_idx][qubit_2_idx] = edge_info["fidelity"]
        distance_matrix[qubit_2_idx][qubit_1_idx] = edge_info["fidelity"]

    # Calculate the strongest path between two qubit.
    for u in qubits_idx:
        for v in qubits_idx:
            for w in qubits_idx:
                tmp_value = distance_matrix[u][v] * distance_matrix[v][w]
                if tmp_value > distance_matrix[u][w]:
                    distance_matrix[u][w] = tmp_value
                    distance_matrix[w][u] = tmp_value
    return distance_matrix
