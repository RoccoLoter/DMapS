import logging
import retworkx
import numpy as np
from bidict import bidict
from copy import copy, deepcopy
from collections import defaultdict

from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass

logger = logging.getLogger(__name__)

EXTENDED_SET_WEIGHT = 0.5  # Weight of lookahead window compared to front_layer.
DECAY_RATE = 0.001  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class MultiModeSwap(TransformationPass):
    def __init__(
        self,
        total_coupling_map,
        local_coupling_map,
        heuristic="lookahead",
        seed=None,
        fake_run=False,
        commu_qubit_layout=None,
        input_layout=None,
        input_matrix_tolist=None,
        phy_qubit_n_idx=None,
        lookahead_ability=20,
    ):
        super().__init__()

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if total_coupling_map is None or total_coupling_map.is_symmetric:
            self.coupling_map = deepcopy(total_coupling_map)
        else:
            self.coupling_map = deepcopy(total_coupling_map)
            self.coupling_map.make_symmetric()

        if local_coupling_map is None or local_coupling_map.is_symmetric:
            self.local_coupling_map = deepcopy(local_coupling_map)
        else:
            self.local_coupling_map = deepcopy(local_coupling_map)
            self.local_coupling_map.make_symmetric()

        self.heuristic = heuristic
        self.seed = seed
        self.fake_run = fake_run
        self.commu_qubit_layout = commu_qubit_layout
        self.input_layout = input_layout
        if input_matrix_tolist is not None:
            self.dist_matrix = np.array(input_matrix_tolist)
        else:
            self.dist_matrix = None
        self.phy_qubit_n_idx = phy_qubit_n_idx
        self.lookahead_ability = lookahead_ability

        self.qubits_decay = None
        self._bit_indices = None
        self.required_predecessors = None
        self.extended_set_size = 20

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
        local_qubits = list(self.input_layout._v2p.keys())
        max_iterations_without_progress = 10 * len(local_qubits)  # Arbitrary.
        ops_since_progress = []
        extended_set = None

        # Normally this isn't necessary, but here we want to log some objects that have some
        # non-trivial cost to create.
        do_expensive_logging = logger.isEnabledFor(logging.DEBUG)
        if self.dist_matrix is None:
            raise TranspilerError("The distance matrix is required.")
        rng = np.random.default_rng(self.seed)

        # Get the canonical register.
        canonical_register = None
        dag_registers = dag.qregs
        if len(dag_registers) == 1:
            for _, reg in dag_registers.items():
                canonical_register = reg
        else:
            for _, reg in dag_registers.items():
                if reg.name == "lq":
                    canonical_register = reg

        current_layout = None
        if self.input_layout is not None:
            current_layout = self.input_layout
        else:
            raise TranspilerError("The initial layout is required.")

        self._bit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}

        # A decay factor for each qubit used to heuristically penalize recently
        # used qubits (to encourage parallelism).
        # self.qubits_decay = dict.fromkeys(local_qubits, 1)

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
                    # Accessing layout._v2p directly to avoid overhead from __getitem__ and a
                    # single access isn't feasible because the layout is updated on each iteration
                    if self._is_executable(node, current_layout):
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
                self._undo_operations(ops_since_progress, current_layout)
                self._add_greedy_swaps(front_layer, current_layout)
                continue

            if execute_gate_list:
                for node in execute_gate_list:
                    for successor in self._successors(node, dag):
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            front_layer.append(successor)

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

                score = self._score_heuristic(
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
            current_layout.swap(*best_swap)
            ops_since_progress.append(best_swap)
            num_search_steps += 1

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

        return current_layout

    def _is_executable(self, op, current_layout):
        """Return True if the node can be executed on the current layout."""
        is_executable = False

        fst_qubit, snd_qubit = op.qargs
        is_remote_op = self._is_remote_op(op)
        if is_remote_op:
            commu_qubit = None
            local_qubit = None
            if fst_qubit in self.commu_qubit_layout:
                commu_qubit = fst_qubit
                local_qubit = snd_qubit

            if snd_qubit in self.commu_qubit_layout:
                commu_qubit = snd_qubit
                local_qubit = fst_qubit

            # fst_idx = self.phy_qubit_n_idx.inverse[self.commu_qubit_layout[commu_qubit]]
            # snd_idx = self.phy_qubit_n_idx.inverse[current_layout._v2p[local_qubit]]
            # if self.coupling_map.graph.has_edge(fst_idx, snd_idx):
            if self.commu_qubit_layout[commu_qubit] == current_layout._v2p[local_qubit]:
                is_executable = True
        else:
            real_fst_phy_qubit = self.phy_qubit_n_idx.inverse[
                current_layout._v2p[fst_qubit]
            ]
            real_snd_phy_qubit = self.phy_qubit_n_idx.inverse[
                current_layout._v2p[snd_qubit]
            ]
            if self.coupling_map.graph.has_edge(real_fst_phy_qubit, real_snd_phy_qubit):
                is_executable = True

        return is_executable

    def _is_remote_op(self, op):
        """Return True if the operation is not a local gate."""
        is_remote = False

        fst_qubit, snd_qubit = op.qargs
        if fst_qubit in self.commu_qubit_layout or snd_qubit in self.commu_qubit_layout:
            is_remote = True

        return is_remote

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
            is_remote_op = self._is_remote_op(node)
            if is_remote_op:
                local_qubit = None
                fst_qubit, snd_qubit = node.qargs

                if fst_qubit in self.commu_qubit_layout:
                    local_qubit = snd_qubit

                if snd_qubit in self.commu_qubit_layout:
                    local_qubit = fst_qubit

                physical = current_layout[local_qubit]
                for neighbor in self.coupling_map.neighbors(
                    self.phy_qubit_n_idx.inverse[physical]
                ):
                    # NewCode: Neighbor qubits can only be found within the range of the quantum processor
                    if neighbor not in self.phy_qubit_n_idx:
                        continue

                    virtual_neighbor = current_layout[self.phy_qubit_n_idx[neighbor]]
                    swap = sorted(
                        [local_qubit, virtual_neighbor],
                        key=lambda q: self._bit_indices[q],
                    )
                    candidate_swaps.add(tuple(swap))
            else:
                for virtual in node.qargs:
                    physical = self.phy_qubit_n_idx.inverse[current_layout[virtual]]
                    for neighbor in self.coupling_map.neighbors(physical):
                        # NewCode: Neighbor qubits can only be found within the range of the quantum processor
                        if neighbor not in self.phy_qubit_n_idx:
                            continue

                        virtual_neighbor = current_layout[
                            self.phy_qubit_n_idx[neighbor]
                        ]
                        swap = sorted(
                            [virtual, virtual_neighbor],
                            key=lambda q: self._bit_indices[q],
                        )
                        candidate_swaps.add(tuple(swap))

        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, layout):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p

        node_scores = {}
        for node in front_layer:
            is_remote_op = self._is_remote_op(node)
            if is_remote_op:
                local_qubit = None
                commu_qubit = None

                if node.qargs[0] in self.commu_qubit_layout:
                    commu_qubit = node.qargs[0]
                    local_qubit = node.qargs[1]

                if node.qargs[1] in self.commu_qubit_layout:
                    commu_qubit = node.qargs[1]
                    local_qubit = node.qargs[0]

                # Get the physical qubit index that corresponds to the qubit distance matrix
                commu_idx = self.phy_qubit_n_idx.inverse[
                    self.commu_qubit_layout[commu_qubit]
                ]
                local_idx = self.phy_qubit_n_idx.inverse[layout_map[local_qubit]]
                node_scores[node] = self.dist_matrix[commu_idx, local_idx] + 1
            else:
                fst_idx = self.phy_qubit_n_idx.inverse[layout_map[node.qargs[0]]]
                snd_idx = self.phy_qubit_n_idx.inverse[layout_map[node.qargs[1]]]
                node_scores[node] = self.dist_matrix[fst_idx, snd_idx]
        sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1])
        target_node = sorted_nodes[0][0]

        for pair in _shortest_swap_path(
            tuple(target_node.qargs),
            self.local_coupling_map,
            layout,
            self.commu_qubit_layout,
            self.phy_qubit_n_idx,
        ):
            layout.swap(*pair)

    def _compute_cost(self, layer, layout):
        cost = 0
        layout_map = layout._v2p
        for node in layer:
            is_remote_op = self._is_remote_op(node)
            if not is_remote_op:
                fst_phy_id = self.phy_qubit_n_idx.inverse[layout_map[node.qargs[0]]]
                snd_phy_id = self.phy_qubit_n_idx.inverse[layout_map[node.qargs[1]]]
                cost += self.dist_matrix[fst_phy_id, snd_phy_id]
            else:
                fst_qubit, snd_qubit = node.qargs
                commu_qubit = None
                local_qubit = None
                if fst_qubit in self.commu_qubit_layout:
                    commu_qubit = fst_qubit
                    local_qubit = snd_qubit

                if snd_qubit in self.commu_qubit_layout:
                    commu_qubit = snd_qubit
                    local_qubit = fst_qubit

                fst_id = self.phy_qubit_n_idx.inverse[
                    self.commu_qubit_layout[commu_qubit]
                ]
                snd_id = self.phy_qubit_n_idx.inverse[layout_map[local_qubit]]
                cost += self.dist_matrix[fst_id, snd_id] + 1

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

        raise TranspilerError("Heuristic %s not recognized." % heuristic)

    def _undo_operations(self, swaps, layout):
        for swap in reversed(swaps):
            layout.swap(*swap)


def _transform_gate_for_layout(op_node, layout, device_qreg):
    """Return node implementing a virtual op on given layout."""
    mapped_op_node = copy(op_node)
    mapped_op_node.qargs = [device_qreg[layout._v2p[x]] for x in op_node.qargs]
    return mapped_op_node


def _shortest_swap_path(
    target_qubits,
    local_coupling_map,
    layout,
    commu_qubit_layout,
    phy_qubit_n_idx: bidict,
):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    forwards, backwards = None, None
    v_start, v_goal = target_qubits

    start, goal = None, None
    if v_start in commu_qubit_layout and v_goal not in commu_qubit_layout:
        start = phy_qubit_n_idx.inverse[layout._v2p[v_goal]]
        goal = phy_qubit_n_idx.inverse[commu_qubit_layout[v_start]]
        path = list(
            retworkx.dijkstra_shortest_paths(
                local_coupling_map.graph, start, target=goal
            )[goal]
        )
        forwards = path[1:]
        for swap in forwards:
            yield v_goal, layout._p2v[phy_qubit_n_idx[swap]]

    elif v_start not in commu_qubit_layout and v_goal in commu_qubit_layout:
        start = phy_qubit_n_idx.inverse[layout._v2p[v_start]]
        goal = phy_qubit_n_idx.inverse[commu_qubit_layout[v_goal]]
        path = list(
            retworkx.dijkstra_shortest_paths(
                local_coupling_map.graph, start, target=goal
            )[goal]
        )
        forwards = path[1:]
        for swap in forwards:
            yield v_start, layout._p2v[phy_qubit_n_idx[swap]]

    else:
        start = phy_qubit_n_idx.inverse[layout._v2p[v_start]]
        goal = phy_qubit_n_idx.inverse[layout._v2p[v_goal]]

        # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
        path = list(
            retworkx.dijkstra_shortest_paths(
                local_coupling_map.graph, start, target=goal
            )[goal]
        )
        # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
        # parallelise and reduce depth.

        split = len(path) // 2
        forwards, backwards = path[1:split], reversed(path[split:-1])

        for swap in forwards:
            yield v_start, layout._p2v[phy_qubit_n_idx[swap]]
        for swap in backwards:
            yield v_goal, layout._p2v[phy_qubit_n_idx[swap]]
