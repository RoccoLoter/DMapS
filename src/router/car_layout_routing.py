import logging
import itertools
import numpy as np
import networkx as nx
from copy import copy, deepcopy
from collections import defaultdict
from typing import List, Tuple, Dict

from qiskit.circuit import Qubit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.full_ancilla_allocation import (
    FullAncillaAllocation,
)

from frontend.chips_info_reader import ChipsNet

logger = logging.getLogger(__name__)

EXTENDED_SET_SIZE = (
    10  # Size of lookahead window. TODO: set dynamically to len(current_layout)
)

EXTENDED_SET_WEIGHT = 0.5  # Weight of routing cost of remote gates in lookahead window compared to front_layer.
EXTENDED_SET_RM_WEIGHT = 0.7

RM_SWAP_WEIGHT = 4
EPR_WEIGHT = 2

DECAY_RATE = 1.2  # Decay coefficient for penalizing serial swaps.
DECAY_RESET_INTERVAL = 5  # How often to reset all decay rates to 1.


class CARLayout(AnalysisPass):
    """Choose a Layout via iterative bidirectional routing of the input circuit."""

    def __init__(
        self,
        coupling_map: CouplingMap,
        chips_net: ChipsNet,
        cost_matrix_tolist: List,
        chip_dist_matrix_tolist: List,
        input_layout=None,
        routing_pass=None,
        seed=None,
        max_iterations=5,
        heuristic="lookahead",
    ):
        super().__init__()

        self.seed = seed
        self.heuristic = heuristic
        self.coupling_map = coupling_map
        self.routing_pass = routing_pass
        self.input_layout = input_layout
        self.max_iterations = max_iterations

        self.cost_matrix_tolist = cost_matrix_tolist
        self.chip_dist_matrix_tolist = chip_dist_matrix_tolist

        # Get the quantum chips network related information.
        self.chips_net = chips_net

    def run(self, dag):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Raises:
            TranspilerError: if dag wider than self.coupling_map
        """
        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)

        # Create the initial_layout.
        initial_layout = None
        if self.input_layout is not None:
            initial_layout = self.input_layout
        else:
            raise TranspilerError("The initial qubits mapping must be provided.")

        # Save the initial mapping result
        initial_mapping_result = deepcopy(initial_layout._v2p)

        # Do forward-backward iterations.
        circ = dag_to_circuit(dag)
        rev_circ = circ.reverse_ops()
        ori_dag = dag
        rev_dag = circuit_to_dag(rev_circ)
        for _ in range(self.max_iterations):
            for _ in ("forward", "backward"):
                router = CARSwap(
                    coupling_map=self.coupling_map,
                    chips_net=self.chips_net,
                    heuristic=self.heuristic,
                    cost_matrix_tolist=self.cost_matrix_tolist,
                    chip_dist_matrix_tolist=self.chip_dist_matrix_tolist,
                    seed=self.seed,
                    fake_run=True,
                    input_layout=initial_layout,
                )
                final_layout, _ = router.run(ori_dag)

                # Update initial layout and reverse the unmapped circuit.
                initial_layout = final_layout
                ori_dag, rev_dag = rev_dag, ori_dag

            # Diagnostics
            logger.info("new initial layout")
            logger.info(initial_layout)

        for qreg in dag.qregs.values():
            initial_layout.add_register(qreg)

        # get final mapping result and mapped circuit
        final_mapping_result = deepcopy(initial_layout._v2p)

        return initial_mapping_result, final_mapping_result, initial_layout

    def _layout_and_route_passmanager(self, initial_layout):
        """Return a passmanager for a full layout and routing.

        We use a factory to remove potential statefulness of passes.
        """
        layout_and_route = [
            SetLayout(initial_layout),
            FullAncillaAllocation(self.coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            self.routing_pass,
        ]
        pm = PassManager(layout_and_route)
        return pm

    def _compose_layouts(self, initial_layout, pass_final_layout, qregs):
        """Return the real final_layout resulting from the composition
        of an initial_layout with the final_layout reported by a pass.

        The routing passes internally start with a trivial layout, as the
        layout gets applied to the circuit prior to running them. So the
        "final_layout" they report must be amended to account for the actual
        initial_layout that was selected.
        """
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {
            v: pass_final_layout._v2p[qubit_map[v]] for v in initial_layout._v2p
        }
        return Layout(final_layout)


class CARSwap(TransformationPass):
    def __init__(
        self,
        coupling_map: CouplingMap,
        chips_net: ChipsNet,
        cost_matrix_tolist: List,
        chip_dist_matrix_tolist: List,
        heuristic="lookahead",
        seed=None,
        fake_run=False,
        input_layout=None,
        intra_chip_all2all=False,
    ):
        super().__init__()

        self.seed = seed
        self.qubits_decay = None
        self._bit_indices = None
        self.fake_run = fake_run
        self.heuristic = heuristic
        self.phy_comm_data_qubits = []
        self.input_layout = input_layout
        self.required_predecessors = None
        self.cost_matrix = np.array(cost_matrix_tolist)
        self.chip_dist_matrix = np.array(chip_dist_matrix_tolist)
        self.weight_graph = chips_net.obtain_weighted_network_graph()
        self.intra_chip_all2all = intra_chip_all2all
        self.property_set["final_layout"] = None
        
        self.qc_n_oqc = []
        self.qc_list = []
        self.cx_id = 0
        self.oqc_list = []

        self.extended_set_size = EXTENDED_SET_SIZE

        # Assume bidirectional couplings, fixing gate direction is easy later.
        if coupling_map is None or coupling_map.is_symmetric:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = deepcopy(coupling_map)
            self.coupling_map.make_symmetric()

        # Get the physical qubits index and its related quantum chip index.
        self.phy_qubits_chips_idx = chips_net.obtain_qubit_n_chip_idx()
        self.chips_idx = chips_net.obtain_chips_idx()

        # Get the total physical communication data qubits in the quantum chip network.
        self.each_chip_commu_qubits = chips_net.get_each_chip_commu_qubits_idx()
        for _, phy_commu_qubits in self.each_chip_commu_qubits.items():
            for qubit in phy_commu_qubits:
                self.phy_comm_data_qubits.append(qubit)

        self.qubits_name_n_idx = chips_net.qubits_n_index

        self.chips_nearest_commu_qubit = _create_commu_rout_info(
            self.chips_idx, self.each_chip_commu_qubits, self.cost_matrix
        )

        self.commu_qubit_used_info = {}
        for chip_pair, commu_qubit_pair_list in self.chips_nearest_commu_qubit.items():
            used_info = {qubit_pair: (0, 0) for qubit_pair in commu_qubit_pair_list}
            self.commu_qubit_used_info[chip_pair] = deepcopy(used_info)

        self.chips_nearest_commu_qubit_list = {}
        for chip_pair, commu_qubit_pair_list in self.chips_nearest_commu_qubit.items():
            fst_commu_qubits = []
            sec_commu_qubits = []
            for qubit_pair in commu_qubit_pair_list:
                fst_commu_qubits.append(qubit_pair[0])
                sec_commu_qubits.append(qubit_pair[1])
            tmp_data = [fst_commu_qubits, sec_commu_qubits]
            self.chips_nearest_commu_qubit_list[chip_pair] = deepcopy(tmp_data)

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

        ops_since_progress = []
        swaps_decay_since_progress = {}
        extended_set = None
        max_iterations_without_progress = 10 * len(dag.qubits)  # Arbitrary.

        mapped_dag = None
        canonical_register = None
        current_layout = None

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Swap runs on physical circuits only.")

        if len(dag.qubits) > self.coupling_map.size():
            raise TranspilerError("More virtual qubits exist than physical.")

        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(self.seed)

        # Normally this isn't necessary, but here we want to log some objects that have some
        # non-trivial cost to create.
        do_expensive_logging = logger.isEnabledFor(logging.DEBUG)

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        if not self.fake_run:
            mapped_dag = dag.copy_empty_like()

        # Get the canonical register.
        dag_registers = dag.qregs
        if len(dag_registers) == 1:
            for _, reg in dag_registers.items():
                canonical_register = reg
        else:
            raise TranspilerError("To be realized.")

        # Update the current layout.
        if self.input_layout is not None:
            current_layout = self.input_layout
        else:
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
                if node.name == "cx" or node.name == "cz":
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
                self._add_greedy_swaps(
                    front_layer, mapped_dag, current_layout, canonical_register
                )

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
                swaps_decay_since_progress = {}
                extended_set = None
                continue

            # After all free gates are exhausted, heuristically find
            # the best swap and insert it. When two or more swaps tie
            # for best score, pick one randomly.
            front_layer_info = self._layer_info(front_layer, current_layout)

            extend_set_info = None
            if extended_set is None:
                extended_set = self._obtain_extended_set(dag, front_layer)
            if extended_set is not None:
                extend_set_info = self._layer_info(extended_set, current_layout)

            if len(front_layer_info["local_ops"]) > 0:
                local_ops = front_layer_info["local_ops"]

                if extended_set is None:
                    extended_set = self._obtain_extended_set(
                        dag=dag,
                        front_layer=front_layer,
                        ops_type="local_ops",
                        ops_info=local_ops,
                        current_layout=current_layout,
                    )
                if extended_set is not None:
                    extend_set_info = self._layer_info(extended_set, current_layout)

                swaps_score = {}
                for candidate_swap in self._obtain_swaps(local_ops, current_layout):
                    is_remote_swap = self._is_remote_swap(
                        candidate_swap, current_layout
                    )

                    if is_remote_swap:
                        continue

                    trial_layout = current_layout.copy()
                    trial_layout.swap(*candidate_swap)
                    score = self._heuristic_score(trial_layout, local_ops, extended_set)
                    swaps_score[candidate_swap] = score

                best_swap = self._select_best_swap(
                    swaps_score, current_layout, swaps_decay_since_progress
                )

                swap_node = self._apply_gate(
                    mapped_dag,
                    DAGOpNode(op=SwapGate(), qargs=best_swap),
                    current_layout,
                    canonical_register,
                )
                current_layout.swap(*best_swap)
                ops_since_progress.append(swap_node)
                self._update_swap_decay(
                    best_swap, current_layout, swaps_decay_since_progress
                )
                num_search_steps += 1

            else:
                if len(front_layer_info["rm_ops"]) > 0:
                    ops = front_layer_info["rm_ops"] + front_layer_info["local_rm_ops"]

                    if extended_set is None:
                        extended_set = self._obtain_extended_set(
                            dag=dag,
                            front_layer=front_layer,
                            ops_type="rm_ops",
                            ops_info=ops,
                            current_layout=current_layout,
                        )
                    if extended_set is not None:
                        extend_set_info = self._layer_info(extended_set, current_layout)

                    rm_ops = front_layer_info["rm_ops"]
                    swap_scores = {}
                    for candidate_swap in self._obtain_swaps(rm_ops, current_layout):
                        is_remote_swap = self._is_remote_swap(
                            candidate_swap, current_layout
                        )
                        if not is_remote_swap:
                            continue

                        trial_layout = current_layout.copy()
                        trial_layout.swap(*candidate_swap)
                        score = self._rm_heuristic_score(
                            current_layout=trial_layout,
                            front_layer=rm_ops,
                            extended_set=extended_set,
                        )
                        swap_scores[candidate_swap] = score

                    best_swap = self._select_best_swap(
                        swap_scores, current_layout, swaps_decay_since_progress
                    )

                    swap_node = self._apply_gate(
                        mapped_dag,
                        DAGOpNode(op=SwapGate(), qargs=best_swap),
                        current_layout,
                        canonical_register,
                    )
                    current_layout.swap(*best_swap)
                    ops_since_progress.append(swap_node)
                    self._update_swap_decay(
                        best_swap, current_layout, swaps_decay_since_progress
                    )
                    num_search_steps += 1
                else:
                    if len(front_layer_info["local_rm_ops"]) > 0:
                        local_rm_ops = front_layer_info["local_rm_ops"]
                        if extended_set is None:
                            extended_set = self._obtain_extended_set(
                                dag=dag,
                                front_layer=front_layer,
                                ops_type="local_rm_ops",
                                ops_info=local_rm_ops,
                                current_layout=current_layout,
                            )
                        if extended_set is not None:
                            extend_set_info = self._layer_info(
                                extended_set, current_layout
                            )

                        (
                            op_occupy_commu_qubit,
                            occupy_ops,
                            each_chip_layer_info,
                        ) = self._each_chip_layer_for_rm(
                            current_layout, front_layer_info, extend_set_info
                        )

                        if op_occupy_commu_qubit:
                            swap_scores = {}

                            adjacent_ops = []
                            layout_mapping = current_layout._v2p
                            for op in occupy_ops:
                                qubits = op.qargs
                                fst_phy_qubit, sec_phy_qubit = (
                                    layout_mapping[qubits[0]],
                                    layout_mapping[qubits[1]],
                                )
                                fst_chip_idx, sec_chip_idx = (
                                    self.phy_qubits_chips_idx[fst_phy_qubit],
                                    self.phy_qubits_chips_idx[sec_phy_qubit],
                                )

                                if (
                                    self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]
                                    == 1
                                ):
                                    adjacent_ops.append(op)
                                else:
                                    continue

                            if len(adjacent_ops) > 0:
                                for candidate_swap in self._obtain_swaps(
                                    adjacent_ops, current_layout
                                ):
                                    is_remote_swap = self._is_remote_swap(
                                        candidate_swap, current_layout
                                    )

                                    trial_layout = current_layout.copy()
                                    trial_layout.swap(*candidate_swap)
                                    score = self._rm_local_heuristic_score(
                                        is_remote_swap=is_remote_swap,
                                        current_layout=trial_layout,
                                        front_layer=adjacent_ops,
                                        extended_set=extended_set,
                                    )
                                    swap_scores[candidate_swap] = score

                            else:
                                for candidate_swap in self._obtain_swaps(
                                    occupy_ops, current_layout
                                ):
                                    is_remote_swap = self._is_remote_swap(
                                        candidate_swap, current_layout
                                    )
                                    if is_remote_swap:
                                        continue

                                    trial_layout = current_layout.copy()
                                    trial_layout.swap(*candidate_swap)
                                    score = self._rm_local_swap_heuristic_score(
                                        current_layout=trial_layout,
                                        front_layer=occupy_ops,
                                        extended_set=extended_set,
                                    )
                                    swap_scores[candidate_swap] = score

                            best_swap = self._select_best_swap(
                                swap_scores,
                                current_layout,
                                swaps_decay_since_progress,
                            )

                            swap_node = self._apply_gate(
                                mapped_dag,
                                DAGOpNode(op=SwapGate(), qargs=best_swap),
                                current_layout,
                                canonical_register,
                            )
                            current_layout.swap(*best_swap)
                            ops_since_progress.append(swap_node)
                            self._update_swap_decay(
                                best_swap,
                                current_layout,
                                swaps_decay_since_progress,
                            )
                            num_search_steps += 1

                        else:
                            inserted_swaps = []
                            for chip_idx, layer_info in each_chip_layer_info.items():
                                tmp_front_layer = list(
                                    layer_info["front_layer"]["local_rm_ops"].keys()
                                )

                                if len(tmp_front_layer) > 0:
                                    swap_scores = {}
                                    for tmp_swap in self._obtain_swaps(
                                        tmp_front_layer, current_layout, chip_idx
                                    ):
                                        is_remote_swap = self._is_remote_swap(
                                            tmp_swap, current_layout
                                        )
                                        if is_remote_swap:
                                            continue

                                        trial_layout = current_layout.copy()
                                        trial_layout.swap(*tmp_swap)

                                        score = self._each_chip_rm_heuristic_score(
                                            chip_idx=chip_idx,
                                            current_layout=trial_layout,
                                            front_layer_info=layer_info["front_layer"],
                                            extended_set_info=layer_info[
                                                "extended_set"
                                            ],
                                        )
                                        swap_scores[tmp_swap] = score

                                    best_swap = self._select_best_swap(
                                        swap_scores,
                                        current_layout,
                                        swaps_decay_since_progress,
                                    )
                                    inserted_swaps.append(best_swap)
                                else:
                                    continue

                            for swap in inserted_swaps:
                                swap_node = self._apply_gate(
                                    mapped_dag,
                                    DAGOpNode(op=SwapGate(), qargs=swap),
                                    current_layout,
                                    canonical_register,
                                )
                                current_layout.swap(*swap)
                                ops_since_progress.append(swap_node)

                                self._update_swap_decay(
                                    swap, current_layout, swaps_decay_since_progress
                                )

                                num_search_steps += 1

        self.property_set["final_layout"] = current_layout
        if not self.fake_run:
            return mapped_dag
        return current_layout, dag

    def _update_swap_decay(self, swap, current_layout, swap_dacay):
        """Update the swap decay factor for each swap operation.

        Args:
            swap (list): The swap operation's qargs.
            swap_dacay (dict): The swap decay factor for each swap operation.
        """
        mapping = current_layout._v2p

        tuple_swap = (mapping[swap[0]], mapping[swap[1]])
        rev_tuple_swap = (mapping[swap[1]], mapping[swap[0]])

        if tuple_swap in swap_dacay and rev_tuple_swap not in swap_dacay:
            swap_dacay[tuple_swap] *= DECAY_RATE
        elif tuple_swap not in swap_dacay and rev_tuple_swap in swap_dacay:
            swap_dacay[rev_tuple_swap] *= DECAY_RATE
        elif tuple_swap not in swap_dacay and rev_tuple_swap not in swap_dacay:
            swap_dacay[tuple_swap] = DECAY_RATE

    def _each_chip_layer(self, current_layout, front_layer_info, extended_set_info):
        mapping = current_layout._v2p

        layer_info = {
            "front_layer": {"local_ops": [], "local_rm_ops": {}, "rm_ops": {}},
            "extended_set": {"local_ops": [], "local_rm_ops": [], "rm_ops": []},
        }
        each_chip_layer_info = {
            chip_idx: deepcopy(layer_info) for chip_idx in self.chips_idx
        }

        if front_layer_info["local_ops"] is not None:
            for op in front_layer_info["local_ops"]:
                fst_phy_qubit = mapping[op.qargs[0]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                each_chip_layer_info[fst_chip_idx]["front_layer"]["local_ops"].append(
                    op
                )

        if front_layer_info["local_rm_ops"] is not None:
            for op in front_layer_info["local_rm_ops"]:
                fst_phy_qubit = mapping[op.qargs[0]]
                sec_phy_qubit = mapping[op.qargs[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                chips_pair = (fst_chip_idx, sec_chip_idx)
                rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                chip_dist = self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]
                if chip_dist > 1:
                    if chips_pair in self.chips_nearest_commu_qubit:
                        qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]

                        is_local_rm_op = True
                        for qubit_pair in qubit_pairs:
                            if (
                                fst_phy_qubit == qubit_pair[0]
                                and sec_phy_qubit != qubit_pair[1]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[0]
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = -1
                                is_local_rm_op = False
                                break

                            elif (
                                fst_phy_qubit != qubit_pair[0]
                                and sec_phy_qubit == qubit_pair[1]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = -1
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[1]
                                is_local_rm_op = False
                                break
                            else:
                                continue

                        if is_local_rm_op:
                            each_chip_layer_info[fst_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][
                                op
                            ] = -1  # -1 means both qubits are not communication qubit.
                            each_chip_layer_info[sec_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][op] = -1

                    elif rev_chips_pair in self.chips_nearest_commu_qubit:
                        qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]

                        is_local_rm_op = True
                        for qubit_pair in qubit_pairs:
                            if (
                                fst_phy_qubit == qubit_pair[1]
                                and sec_phy_qubit != qubit_pair[0]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[1]
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = -1
                                is_local_rm_op = False
                                break

                            elif (
                                fst_phy_qubit != qubit_pair[1]
                                and sec_phy_qubit == qubit_pair[0]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = -1
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[0]
                                is_local_rm_op = False
                                break

                            else:
                                continue

                        if is_local_rm_op:
                            each_chip_layer_info[fst_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][op] = -1
                            each_chip_layer_info[sec_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][op] = -1

                elif chip_dist == 1:
                    if chips_pair in self.chips_nearest_commu_qubit:
                        qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]

                        is_local_rm_op = True
                        for qubit_pair in qubit_pairs:
                            if (
                                fst_phy_qubit == qubit_pair[0]
                                and sec_phy_qubit != qubit_pair[1]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[0]
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[1]
                                is_local_rm_op = False
                                break

                            elif (
                                fst_phy_qubit != qubit_pair[0]
                                and sec_phy_qubit == qubit_pair[1]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[0]
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[1]
                                is_local_rm_op = False
                                break
                            else:
                                continue

                        if is_local_rm_op:
                            each_chip_layer_info[fst_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][
                                op
                            ] = -1  # -1 means both qubits are not communication qubit.
                            each_chip_layer_info[sec_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][op] = -1

                    elif rev_chips_pair in self.chips_nearest_commu_qubit:
                        qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]

                        is_local_rm_op = True
                        for qubit_pair in qubit_pairs:
                            if (
                                fst_phy_qubit == qubit_pair[1]
                                and sec_phy_qubit != qubit_pair[0]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[1]
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[0]
                                is_local_rm_op = False
                                break

                            elif (
                                fst_phy_qubit != qubit_pair[1]
                                and sec_phy_qubit == qubit_pair[0]
                            ):
                                each_chip_layer_info[fst_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[1]
                                each_chip_layer_info[sec_chip_idx]["front_layer"][
                                    "local_rm_ops"
                                ][op] = qubit_pair[0]
                                is_local_rm_op = False
                                break

                            else:
                                continue

                        if is_local_rm_op:
                            each_chip_layer_info[fst_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][op] = -1
                            each_chip_layer_info[sec_chip_idx]["front_layer"][
                                "local_rm_ops"
                            ][op] = -1

        if front_layer_info["rm_ops"] is not None:
            for op in front_layer_info["rm_ops"]:
                fst_phy_qubit = mapping[op.qargs[0]]
                sec_phy_qubit = mapping[op.qargs[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                each_chip_layer_info[fst_chip_idx]["front_layer"]["rm_ops"][
                    op
                ] = fst_phy_qubit
                each_chip_layer_info[sec_chip_idx]["front_layer"]["rm_ops"][
                    op
                ] = sec_phy_qubit

        if extended_set_info["local_ops"] is not None:
            for op in extended_set_info["local_ops"]:
                fst_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[0]]]
                each_chip_layer_info[fst_chip_idx]["extended_set"]["local_ops"].append(
                    op
                )

        if extended_set_info["local_rm_ops"] is not None:
            for op in extended_set_info["local_rm_ops"]:
                fst_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[0]]]
                sec_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[1]]]
                each_chip_layer_info[fst_chip_idx]["extended_set"][
                    "local_rm_ops"
                ].append(op)
                each_chip_layer_info[sec_chip_idx]["extended_set"][
                    "local_rm_ops"
                ].append(op)

        if extended_set_info["rm_ops"] is not None:
            for op in extended_set_info["rm_ops"]:
                fst_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[0]]]
                sec_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[1]]]
                each_chip_layer_info[fst_chip_idx]["extended_set"]["rm_ops"].append(op)
                each_chip_layer_info[sec_chip_idx]["extended_set"]["rm_ops"].append(op)

        return each_chip_layer_info

    def _each_chip_layer_for_rm(
        self, current_layout, front_layer_info, extended_set_info
    ):
        mapping = current_layout._v2p

        layer_info = {
            "front_layer": {"local_rm_ops": {}},
            "extended_set": {"local_ops": [], "local_rm_ops": [], "rm_ops": []},
        }
        each_chip_layer_info = {
            chip_idx: deepcopy(layer_info) for chip_idx in self.chips_idx
        }

        occupy_ops = []
        op_occupy_commu_qubit = False
        if front_layer_info["local_rm_ops"] is not None:
            for op in front_layer_info["local_rm_ops"]:
                fst_phy_qubit = mapping[op.qargs[0]]
                sec_phy_qubit = mapping[op.qargs[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]
                chips_pair = (fst_chip_idx, sec_chip_idx)
                rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                is_occupy = False
                if chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]
                    for qubit_pair in qubit_pairs:
                        if (
                            fst_phy_qubit == qubit_pair[0]
                            or sec_phy_qubit == qubit_pair[1]
                        ):
                            is_occupy = True
                            occupy_ops.append(op)
                            break
                        else:
                            continue

                elif rev_chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]
                    for qubit_pair in qubit_pairs:
                        if (
                            fst_phy_qubit == qubit_pair[1]
                            or sec_phy_qubit == qubit_pair[0]
                        ):
                            is_occupy = True
                            occupy_ops.append(op)
                            break
                        else:
                            continue

                if not is_occupy:
                    each_chip_layer_info[fst_chip_idx]["front_layer"]["local_rm_ops"][
                        op
                    ] = -1
                    each_chip_layer_info[sec_chip_idx]["front_layer"]["local_rm_ops"][
                        op
                    ] = -1
                else:
                    op_occupy_commu_qubit = True

        if not op_occupy_commu_qubit:
            if extended_set_info["local_ops"] is not None:
                for op in extended_set_info["local_ops"]:
                    fst_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[0]]]
                    each_chip_layer_info[fst_chip_idx]["extended_set"][
                        "local_ops"
                    ].append(op)

            if extended_set_info["local_rm_ops"] is not None:
                for op in extended_set_info["local_rm_ops"]:
                    fst_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[0]]]
                    sec_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[1]]]
                    each_chip_layer_info[fst_chip_idx]["extended_set"][
                        "local_rm_ops"
                    ].append(op)
                    each_chip_layer_info[sec_chip_idx]["extended_set"][
                        "local_rm_ops"
                    ].append(op)

            if extended_set_info["rm_ops"] is not None:
                for op in extended_set_info["rm_ops"]:
                    fst_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[0]]]
                    sec_chip_idx = self.phy_qubits_chips_idx[mapping[op.qargs[1]]]
                    each_chip_layer_info[fst_chip_idx]["extended_set"]["rm_ops"].append(
                        op
                    )
                    each_chip_layer_info[sec_chip_idx]["extended_set"]["rm_ops"].append(
                        op
                    )

        return op_occupy_commu_qubit, occupy_ops, each_chip_layer_info

    def _layer_info(self, current_layer, current_layout):
        layer_info = {}

        local_ops = []
        local_rm_ops = []
        rm_ops = []

        for op in current_layer:
            if op.name == "cx" or op.name == "cz":
                fst_phy_qubit = current_layout[op.qargs[0]]
                sec_phy_qubit = current_layout[op.qargs[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                if fst_chip_idx != sec_chip_idx:
                    chip_dist = self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]
                    if chip_dist == 1:
                        local_rm_ops.append(op)
                    else:
                        chips_pair = (fst_chip_idx, sec_chip_idx)
                        rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                        if chips_pair in self.chips_nearest_commu_qubit_list:
                            each_chip_commu_qubits = (
                                self.chips_nearest_commu_qubit_list[chips_pair]
                            )

                            if (
                                fst_phy_qubit in each_chip_commu_qubits[0]
                                and sec_phy_qubit in each_chip_commu_qubits[1]
                            ):
                                rm_ops.append(op)
                            else:
                                local_rm_ops.append(op)
                        elif rev_chips_pair in self.chips_nearest_commu_qubit_list:
                            each_chip_commu_qubits = (
                                self.chips_nearest_commu_qubit_list[rev_chips_pair]
                            )

                            if (
                                fst_phy_qubit in each_chip_commu_qubits[1]
                                and sec_phy_qubit in each_chip_commu_qubits[0]
                            ):
                                rm_ops.append(op)
                            else:
                                local_rm_ops.append(op)
                else:
                    local_ops.append(op)

        layer_info["local_ops"] = local_ops
        layer_info["local_rm_ops"] = local_rm_ops
        layer_info["rm_ops"] = rm_ops

        return layer_info

    def _apply_gate(self, mapped_dag, node, current_layout, canonical_register):
        new_node = _transform_gate_for_layout(node, current_layout, canonical_register)

        if node.op.name == "cx":
            self.qc_list.append((node.qargs[0].index, node.qargs[1].index, self.cx_id))
            self.oqc_list.append((new_node.qargs[0].index, new_node.qargs[1].index, self.cx_id))
            self.cx_id += 1
        elif new_node.op.name == "swap":
            self.oqc_list.append((new_node.qargs[0].index, new_node.qargs[1].index, -1))

        # self.qc_n_oqc.append((node, new_node))

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

    def _obtain_extended_set(
        self, dag, front_layer, ops_type=None, ops_info=None, current_layout=None
    ):
        """Populate extended_set by looking ahead a fixed number of gates.
        For each existing element add a successor until reaching limit."""
        extended_set = []
        if ops_type is None:
            self.extended_set_size = EXTENDED_SET_SIZE * len(front_layer)

            done = False
            decremented = []
            tmp_front_layer = front_layer
            while tmp_front_layer and not done:
                new_tmp_front_layer = []
                for node in tmp_front_layer:
                    for successor in self._successors(node, dag):
                        decremented.append(successor)
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            new_tmp_front_layer.append(successor)
                            if successor.name == "cx" or successor.name == "cz":
                                extended_set.append(successor)

                    if len(extended_set) >= self.extended_set_size:
                        done = True
                        break

                tmp_front_layer = new_tmp_front_layer

            for node in decremented:
                self.required_predecessors[node] += 1
        else:
            num_ops = 0
            extended_set_size = EXTENDED_SET_SIZE * len(ops_info)

            done = False
            decremented = []
            tmp_front_layer = front_layer
            layout_mapping = current_layout._v2p
            while tmp_front_layer and not done:
                new_tmp_front_layer = []
                for node in tmp_front_layer:
                    for successor in self._successors(node, dag):
                        decremented.append(successor)
                        self.required_predecessors[successor] -= 1
                        if self._is_resolved(successor):
                            new_tmp_front_layer.append(successor)
                            if successor.name == "cx" or successor.name == "cz":
                                qubits = successor.qargs
                                fst_phy_qubit, sec_phy_qubit = (
                                    layout_mapping[qubits[0]],
                                    layout_mapping[qubits[1]],
                                )
                                fst_chip_idx, sec_chip_idx = (
                                    self.phy_qubits_chips_idx[fst_phy_qubit],
                                    self.phy_qubits_chips_idx[sec_phy_qubit],
                                )

                                if ops_type == "local_ops":
                                    if fst_chip_idx == sec_chip_idx:
                                        num_ops += 1
                                        extended_set.append(successor)
                                    else:
                                        extended_set.append(successor)
                                else:
                                    if fst_chip_idx != sec_chip_idx:
                                        num_ops += 1
                                        extended_set.append(successor)
                                    else:
                                        extended_set.append(successor)

                    if num_ops >= extended_set_size:
                        done = True
                        break

                tmp_front_layer = new_tmp_front_layer

            for node in decremented:
                self.required_predecessors[node] += 1

        return extended_set

    def _obtain_swaps(self, front_layer, current_layout, chip_index=None):
        """Return a set of candidate swaps that affect qubits in front_layer.

        For each virtual qubit in front_layer, find its current location
        on hardware and the physical qubits in that neighborhood. Every SWAP
        on virtual qubits that corresponds to one of those physical couplings
        is a candidate SWAP.

        Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
        """
        candidate_swaps = set()

        if chip_index is not None:
            for node in front_layer:
                for virtual in node.qargs:
                    physical = current_layout[virtual]

                    if self.phy_qubits_chips_idx[physical] == chip_index:
                        for neighbor in self.coupling_map.neighbors(physical):
                            virtual_neighbor = current_layout[neighbor]
                            swap = sorted(
                                [virtual, virtual_neighbor],
                                key=lambda q: self._bit_indices[q],
                            )
                            candidate_swaps.add(tuple(swap))
                    else:
                        continue
        else:
            for node in front_layer:
                for virtual in node.qargs:
                    physical = current_layout[virtual]
                    for neighbor in self.coupling_map.neighbors(physical):
                        virtual_neighbor = current_layout[neighbor]
                        swap = sorted(
                            [virtual, virtual_neighbor],
                            key=lambda q: self._bit_indices[q],
                        )
                        candidate_swaps.add(tuple(swap))

        return candidate_swaps

    def _add_greedy_swaps(self, front_layer, dag, layout, qubits):
        """Mutate ``dag`` and ``layout`` by applying greedy swaps to ensure that at least one gate
        can be routed."""
        layout_map = layout._v2p
        op_estimated_cost = {}

        for node in front_layer:
            fst_phy_qubit = layout_map[node.qargs[0]]
            sec_phy_qubit = layout_map[node.qargs[1]]
            fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
            sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

            cost = 0
            if fst_chip_idx != sec_chip_idx:
                chip_dist = self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]
                cost = (
                    self.cost_matrix[fst_phy_qubit][sec_phy_qubit]
                    + (2 * chip_dist - 1) * EPR_WEIGHT
                )
            else:
                cost = self.cost_matrix[fst_phy_qubit][sec_phy_qubit]

            op_estimated_cost[node] = cost

        op_order = sorted(op_estimated_cost.items(), key=lambda x: x[1], reverse=False)
        target_node = op_order[0][0]

        for pair in _shortest_swap_path(
            tuple(target_node.qargs),
            self.weight_graph,
            layout,
            self.phy_qubits_chips_idx,
            self.phy_comm_data_qubits,
        ):
            self._apply_gate(dag, DAGOpNode(op=SwapGate(), qargs=pair), layout, qubits)
            layout.swap(*pair)

    def _heuristic_score(
        self,
        current_layout: Layout,
        front_layer: List[DAGOpNode],
        extended_set: List[DAGOpNode],
    ):
        """Return a heuristic cost score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a distance value
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        total_cost = 0

        fl_cost = self._calculate_cost(front_layer, current_layout)
        total_cost = fl_cost / len(front_layer)

        if len(extended_set) > 0:
            es_cost = self._calculate_cost(extended_set, current_layout)
            final_es_cost = (EXTENDED_SET_WEIGHT * es_cost) / len(extended_set)
            total_cost += final_es_cost

        return total_cost

    def _calculate_cost(
        self,
        current_layer: List[DAGOpNode],
        current_layout: Layout,
    ):
        """Estimate the total EPR pairs usage of the current layout."""
        total_cost = 0

        couping_map = current_layout._v2p
        if len(current_layer) > 0:
            for node in current_layer:
                if node.name == "cx" or node.name == "cz":
                    qubits = node.qargs
                    fst_phy_qubit = couping_map[qubits[0]]
                    sec_phy_qubit = couping_map[qubits[1]]
                    total_cost += self.cost_matrix[fst_phy_qubit][sec_phy_qubit]

        return total_cost

    def _local_heuristic_score(
        self,
        chip_idx: int,
        current_layout: Layout,
        front_layer_info: Dict,
        extended_set_info: Dict,
    ):
        """Return a heuristic cost score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a distance value
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        total_cost = 0

        fl_num_ops = (
            len(front_layer_info["local_ops"])
            + len(front_layer_info["local_rm_ops"])
            + len(front_layer_info["rm_ops"])
        )
        fl_cost = self._fl_calculate_cost_for_local(
            chip_idx, front_layer_info, current_layout
        )
        total_cost = fl_cost / fl_num_ops

        es_num_ops = (
            len(extended_set_info["local_ops"])
            + len(extended_set_info["local_rm_ops"])
            + len(extended_set_info["rm_ops"])
        )
        if es_num_ops > 0:
            es_cost = self._es_calculate_cost(
                chip_idx, extended_set_info, current_layout
            )
            final_es_cost = (EXTENDED_SET_WEIGHT * es_cost) / es_num_ops
            total_cost += final_es_cost

        return total_cost

    def _rm_local_heuristic_score(
        self,
        is_remote_swap: bool,
        current_layout: Layout,
        front_layer: List[DAGOpNode],
        extended_set: List[DAGOpNode],
    ):
        """Return a heuristic cost score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a distance value
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        total_cost = 0

        after_fl_cost, after_fl_eprs = self._rm_local_calculate_cost(
            current_layout, front_layer
        )
        total_cost = after_fl_cost + after_fl_eprs * EPR_WEIGHT

        if is_remote_swap:
            total_cost += RM_SWAP_WEIGHT
        else:
            total_cost += 1

        if len(extended_set) > 0:
            after_es_cost, after_es_eprs = self._rm_local_calculate_cost(
                current_layout, extended_set
            )

            es_cost = after_es_cost + after_es_eprs * EPR_WEIGHT
            final_es_cost = EXTENDED_SET_RM_WEIGHT * es_cost
            total_cost += final_es_cost

        return total_cost

    def _rm_local_swap_heuristic_score(
        self,
        current_layout: Layout,
        front_layer: List[DAGOpNode],
        extended_set: List[DAGOpNode],
    ):
        """Return a heuristic cost score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a distance value
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        total_cost = 0

        fl_cost, _ = self._rm_local_calculate_cost(current_layout, front_layer)
        total_cost = fl_cost / len(front_layer)

        if len(extended_set) > 0:
            es_cost, _ = self._rm_local_calculate_cost(current_layout, extended_set)
            final_es_cost = (EXTENDED_SET_WEIGHT * es_cost) / len(extended_set)
            total_cost += final_es_cost

        return total_cost

    def _each_chip_rm_heuristic_score(
        self,
        chip_idx: int,
        current_layout: Layout,
        front_layer_info: Dict[str, Dict],
        extended_set_info: Dict[str, Dict],
    ):
        """Return a heuristic cost score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a distance value
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        total_cost = 0

        fl_num_ops = len(front_layer_info["local_rm_ops"])
        fl_cost = self._fl_calculate_cost_for_rm(
            chip_idx=chip_idx,
            layer_info=front_layer_info,
            current_layout=current_layout,
        )
        total_cost = fl_cost / fl_num_ops

        es_num_ops = (
            len(extended_set_info["local_ops"])
            + len(extended_set_info["local_rm_ops"])
            + len(extended_set_info["rm_ops"])
        )
        if es_num_ops > 0:
            es_cost = self._es_calculate_cost(
                chip_idx=chip_idx,
                layer_info=extended_set_info,
                current_layout=current_layout,
            )
            final_es_cost = (EXTENDED_SET_WEIGHT * es_cost) / es_num_ops
            total_cost += final_es_cost

        return total_cost

    def _rm_heuristic_score(
        self,
        current_layout: Layout,
        front_layer: List[DAGOpNode],
        extended_set: List[DAGOpNode],
    ):
        """Return a heuristic cost score for a trial layout.

        Assuming a trial layout has resulted from a SWAP, we now assign a distance value
        to it. The goodness of a layout is evaluated based on how viable it makes
        the remaining virtual gates that must be applied.
        """
        total_cost = 0

        fl_cost = self._rm_calculate_cost(front_layer, current_layout)
        total_cost = fl_cost / len(front_layer)

        if len(extended_set) > 0:
            es_cost = self._rm_calculate_cost(extended_set, current_layout)
            final_es_cost = (EXTENDED_SET_RM_WEIGHT * es_cost) / len(extended_set)
            total_cost += final_es_cost

        return total_cost

    def _fl_calculate_cost_for_rm(
        self,
        chip_idx: int,
        current_layout: Layout,
        layer_info: Dict[str, Dict[DAGOpNode, int]],
    ):
        """Calculate the total cost value of the current layout."""
        total_value = 0
        couping_map = current_layout._v2p

        local_rm_ops = layer_info["local_rm_ops"]
        if len(local_rm_ops) > 0:
            for op, _ in local_rm_ops.items():
                qubits = op.qargs
                fst_phy_qubit = couping_map[qubits[0]]
                sec_phy_qubit = couping_map[qubits[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                chips_pair = (fst_chip_idx, sec_chip_idx)
                rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                if chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]

                    tmp_values = []
                    if fst_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[fst_phy_qubit][qubit_pair[0]]
                            tmp_values.append(tmp_value)
                    elif sec_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[sec_phy_qubit][qubit_pair[1]]
                            tmp_values.append(tmp_value)

                    total_value += min(tmp_values)

                elif rev_chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]

                    tmp_values = []
                    if fst_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[fst_phy_qubit][qubit_pair[1]]
                            tmp_values.append(tmp_value)
                    elif sec_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[sec_phy_qubit][qubit_pair[0]]
                            tmp_values.append(tmp_value)

                    total_value += min(tmp_values)

        return total_value

    def _fl_calculate_cost_for_local(
        self,
        chip_idx: int,
        layer_info: Dict[str, List[DAGOpNode]],
        current_layout: Layout,
    ):
        """Calculate the total cost value of the current layout."""
        total_value = 0
        couping_map = current_layout._v2p

        local_ops = layer_info["local_ops"]
        local_rm_ops = layer_info["local_rm_ops"]
        rm_ops = layer_info["rm_ops"]

        if len(local_ops) > 0:
            for op in local_ops:
                qubits = op.qargs
                fst_phy_qubit = couping_map[qubits[0]]
                sec_phy_qubit = couping_map[qubits[1]]

                total_value += self.cost_matrix[fst_phy_qubit][sec_phy_qubit]

        if len(local_rm_ops) > 0:
            for op, comm_qubit in local_rm_ops.items():
                if comm_qubit >= -1:
                    qubits = op.qargs
                    fst_phy_qubit = couping_map[qubits[0]]
                    sec_phy_qubit = couping_map[qubits[1]]
                    fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                    sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                    if comm_qubit >= 0:
                        if fst_chip_idx == chip_idx:
                            total_value += self.cost_matrix[fst_phy_qubit][comm_qubit]

                        if sec_chip_idx == chip_idx:
                            total_value += self.cost_matrix[sec_phy_qubit][comm_qubit]
                    elif comm_qubit == -1:
                        chips_pair = (fst_chip_idx, sec_chip_idx)
                        rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                        if chips_pair in self.chips_nearest_commu_qubit:
                            qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]

                            tmp_values = []
                            if fst_chip_idx == chip_idx:
                                for qubit_pair in qubit_pairs:
                                    tmp_value = self.cost_matrix[fst_phy_qubit][
                                        qubit_pair[0]
                                    ]
                                    tmp_values.append(tmp_value)
                            elif sec_chip_idx == chip_idx:
                                for qubit_pair in qubit_pairs:
                                    tmp_value = self.cost_matrix[sec_phy_qubit][
                                        qubit_pair[1]
                                    ]
                                    tmp_values.append(tmp_value)

                            total_value += min(tmp_values)

                        elif rev_chips_pair in self.chips_nearest_commu_qubit:
                            qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]

                            tmp_values = []
                            if fst_chip_idx == chip_idx:
                                for qubit_pair in qubit_pairs:
                                    tmp_value = self.cost_matrix[fst_phy_qubit][
                                        qubit_pair[1]
                                    ]
                                    tmp_values.append(tmp_value)
                            elif sec_chip_idx == chip_idx:
                                for qubit_pair in qubit_pairs:
                                    tmp_value = self.cost_matrix[sec_phy_qubit][
                                        qubit_pair[0]
                                    ]
                                    tmp_values.append(tmp_value)

                            total_value += min(tmp_values)

        if len(rm_ops) > 0:
            for op, comm_qubit in rm_ops.items():
                qubits = op.qargs
                fst_phy_qubit = couping_map[qubits[0]]
                sec_phy_qubit = couping_map[qubits[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                if fst_chip_idx == chip_idx:
                    total_value += self.cost_matrix[fst_phy_qubit][comm_qubit]

                if sec_chip_idx == chip_idx:
                    total_value += self.cost_matrix[sec_phy_qubit][comm_qubit]

        return total_value

    def _es_calculate_cost(
        self,
        chip_idx: int,
        layer_info: Dict[str, List[DAGOpNode]],
        current_layout: Layout,
    ):
        """Calculate the total cost value of the current layout."""
        total_value = 0

        couping_map = current_layout._v2p
        local_ops = layer_info["local_ops"]
        local_rm_ops = layer_info["local_rm_ops"]
        rm_ops = layer_info["rm_ops"]

        if len(local_ops) > 0:
            for node in local_ops:
                qubits = node.qargs
                fst_phy_qubit = couping_map[qubits[0]]
                sec_phy_qubit = couping_map[qubits[1]]

                total_value += self.cost_matrix[fst_phy_qubit][sec_phy_qubit]

        if len(local_rm_ops) > 0:
            for node in local_rm_ops:
                qubits = node.qargs
                fst_phy_qubit = couping_map[qubits[0]]
                sec_phy_qubit = couping_map[qubits[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                chips_pair = (fst_chip_idx, sec_chip_idx)
                rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                if chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]

                    tmp_values = []
                    if fst_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[fst_phy_qubit][qubit_pair[0]]
                            tmp_values.append(tmp_value)
                    elif sec_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[sec_phy_qubit][qubit_pair[1]]
                            tmp_values.append(tmp_value)

                    total_value += min(tmp_values)

                elif rev_chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]

                    tmp_values = []
                    if fst_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[fst_phy_qubit][qubit_pair[1]]
                            tmp_values.append(tmp_value)
                    elif sec_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[sec_phy_qubit][qubit_pair[0]]
                            tmp_values.append(tmp_value)

                    total_value += min(tmp_values)

        if len(rm_ops) > 0:
            for node in rm_ops:
                qubits = node.qargs
                fst_phy_qubit = couping_map[qubits[0]]
                sec_phy_qubit = couping_map[qubits[1]]
                fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                chips_pair = (fst_chip_idx, sec_chip_idx)
                rev_chips_pair = (sec_chip_idx, fst_chip_idx)

                if chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[chips_pair]

                    tmp_values = []
                    if fst_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[fst_phy_qubit][qubit_pair[0]]
                            tmp_values.append(tmp_value)
                    elif sec_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[sec_phy_qubit][qubit_pair[1]]
                            tmp_values.append(tmp_value)
                    min_value = min(tmp_values)

                    total_value += min_value

                elif rev_chips_pair in self.chips_nearest_commu_qubit:
                    qubit_pairs = self.chips_nearest_commu_qubit[rev_chips_pair]

                    tmp_values = []
                    if fst_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[fst_phy_qubit][qubit_pair[1]]
                            tmp_values.append(tmp_value)
                    elif sec_chip_idx == chip_idx:
                        for qubit_pair in qubit_pairs:
                            tmp_value = self.cost_matrix[sec_phy_qubit][qubit_pair[0]]
                            tmp_values.append(tmp_value)
                    min_value = min(tmp_values)

                    total_value += min_value

        return total_value

    def _rm_calculate_cost(
        self,
        current_layer: List[DAGOpNode],
        current_layout: Layout,
    ):
        """Estimate the total EPR pairs usage of the current layout."""
        total_cost = 0

        total_eprs = 0
        total_dist = 0

        couping_map = current_layout._v2p
        if len(current_layer) > 0:
            for node in current_layer:
                if node.name == "cx" or node.name == "cz":
                    qubits = node.qargs
                    fst_phy_qubit = couping_map[qubits[0]]
                    sec_phy_qubit = couping_map[qubits[1]]
                    fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                    sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                    if fst_chip_idx != sec_chip_idx:
                        chip_dist = self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]
                        total_eprs += 2 * chip_dist - 1

                    total_dist += self.cost_matrix[fst_phy_qubit][sec_phy_qubit]

        total_cost = total_dist + total_eprs * EPR_WEIGHT

        return total_cost

    def _rm_local_calculate_cost(
        self,
        current_layout: Layout,
        current_layer: List[DAGOpNode],
    ):
        """Estimate the total EPR pairs usage of the current layout."""
        total_dist = 0
        total_eprs = 0

        couping_map = current_layout._v2p
        if len(current_layer) > 0:
            for node in current_layer:
                if node.name == "cx" or node.name == "cz":
                    qubits = node.qargs
                    fst_phy_qubit = couping_map[qubits[0]]
                    sec_phy_qubit = couping_map[qubits[1]]
                    fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
                    sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

                    if fst_chip_idx != sec_chip_idx:
                        chip_dist = self.chip_dist_matrix[fst_chip_idx][sec_chip_idx]
                        total_eprs += 2 * chip_dist - 1

                    total_dist += self.cost_matrix[fst_phy_qubit][sec_phy_qubit]

        return total_dist, total_eprs

    def _serves_for_lookahead(self, dag, front_layer, current_layout):
        extended_set_size = 0
        required_predecessors = {
            op: value for op, value in self.required_predecessors.items()
        }

        # ceate a new front layer
        execute_gate_list = []
        new_front_layer = []
        for node in front_layer:
            if node.name == "cx" or node.name == "cz":
                v0, v1 = node.qargs
                if self.coupling_map.graph.has_edge(
                    current_layout._v2p[v0], current_layout._v2p[v1]
                ):
                    execute_gate_list.append(node)
                else:
                    new_front_layer.append(node)
            else:  # Single-qubit gates as well as barriers are free
                execute_gate_list.append(node)

        if execute_gate_list:
            for node in execute_gate_list:
                for successor in self._successors(node, dag):
                    required_predecessors[successor] -= 1
                    if required_predecessors[successor] == 0:
                        new_front_layer.append(successor)

        # create a new extended set
        new_extended_set = []
        done = False
        tmp_front_layer = new_front_layer
        extended_set_size = EXTENDED_SET_SIZE * len(new_front_layer)
        while tmp_front_layer and not done:
            new_tmp_front_layer = []
            for node in tmp_front_layer:
                for successor in self._successors(node, dag):
                    required_predecessors[successor] -= 1
                    if required_predecessors[successor] == 0:
                        new_tmp_front_layer.append(successor)
                        if successor.name == "cx" or successor.name == "cz":
                            new_extended_set.append(successor)

                if len(new_extended_set) >= extended_set_size:
                    done = True
                    break

            tmp_front_layer = new_tmp_front_layer

        return new_front_layer, new_extended_set

    def _select_best_swap(
        self,
        swap_scores: Dict,
        current_layout: Layout,
        swaps_qargs_since_progress: List,
    ):
        """When there are multiple swaps with the same score, select the one that is not remote."""
        best_swap = None

        current_mapping = current_layout._v2p
        for swap, _ in swap_scores.items():
            tuple_qubit = (current_mapping[swap[0]], current_mapping[swap[1]])
            rev_tuple_qubit = (current_mapping[swap[1]], current_mapping[swap[0]])

            if tuple_qubit in swaps_qargs_since_progress:
                swap_scores[swap] *= swaps_qargs_since_progress[tuple_qubit]
            elif rev_tuple_qubit in swaps_qargs_since_progress:
                swap_scores[swap] *= swaps_qargs_since_progress[rev_tuple_qubit]

        sorted_swaps = sorted(swap_scores.items(), key=lambda x: x[1], reverse=False)
        best_swap = sorted_swaps[0][0]

        return best_swap

    def _is_remote_swap(self, swap_qubits: Tuple[Qubit], current_layout: Layout):
        """Judge whether the inserted swap operation is remote."""
        is_remote = False

        current_mapping = current_layout._v2p

        fst_phy_qubit = current_mapping[swap_qubits[0]]
        sec_phy_qubit = current_mapping[swap_qubits[1]]
        fst_chip_idx = self.phy_qubits_chips_idx[fst_phy_qubit]
        sec_chip_idx = self.phy_qubits_chips_idx[sec_phy_qubit]

        if fst_chip_idx != sec_chip_idx:
            is_remote = True
        else:
            is_remote = False

        return is_remote

    def _undo_operations(self, operations, dag: DAGCircuit, layout):
        """Mutate ``dag`` and ``layout`` by undoing the swap gates listed in ``operations``."""
        if dag is None:
            for operation in reversed(operations):
                layout.swap(*operation.qargs)
        else:
            # For code verify
            for operation in reversed(operations):
                reversed_oqc = self.oqc_list[::-1]
                i = 0
                while i < len(reversed_oqc):
                    id_pair = (operation.qregs[0].index, operation.qregs[1].index, -1)
                    if id_pair == reversed_oqc[i]:
                        del reversed_oqc[i]
                        self.oqc_list = deepcopy(reversed_oqc[::-1])
                        break
                    else:
                        i += 1

            for operation in reversed(operations):
                dag.remove_op_node(operation)
                p0 = self._bit_indices[operation.qargs[0]]
                p1 = self._bit_indices[operation.qargs[1]]
                layout.swap(p0, p1)

    def _conv_mapping_res(self, ori_mapping_res, qubits_name_n_idx):
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


def _shortest_swap_path(
    target_qubits,
    weighted_graph,
    layout,
    phy_qubits_chip_info,
    phy_commu_data_qubits,
):
    """Return an iterator that yields the swaps between virtual qubits needed to bring the two
    virtual qubits in ``target_qubits`` together in the coupling map."""
    v_start, v_goal = target_qubits
    start, goal = layout._v2p[v_start], layout._v2p[v_goal]

    fst_chip_idx = phy_qubits_chip_info[start]
    sec_chip_idx = phy_qubits_chip_info[goal]

    # TODO: remove the list call once using retworkx 0.12, as the return value can be sliced.
    path = list(
        nx.dijkstra_path(weighted_graph, source=start, target=goal, weight="weight")
    )

    split = 1
    split_qubit = None
    rev_path = reversed(path)
    if fst_chip_idx != sec_chip_idx:
        for phy_qubit in rev_path:
            if phy_qubit in phy_commu_data_qubits:
                split_qubit = phy_qubit
                break
        for index in range(len(path)):
            if path[index] == split_qubit:
                split = index
                break
    else:
        # Swap both qubits towards the "centre" (as opposed to applying the same swaps to one) to
        # parallelise and reduce depth.
        split = len(path) // 2

    forwards, backwards = path[1:split], reversed(path[split:-1])
    for swap in forwards:
        yield v_start, layout._p2v[swap]
    for swap in backwards:
        yield v_goal, layout._p2v[swap]


def _create_commu_rout_info(
    chips_idx, each_chip_commu_qubits, cost_matrix
) -> Dict[Tuple[int], Tuple[int]]:
    """
    Calculate the distance value of the most nearest path between two quantum chips, and record the information of communication physical qubits at both ends of the path.
    """
    chips_nearest_commu_qubit = {}

    for chip_idx_pair in itertools.combinations(chips_idx, 2):
        fst_chip_idx = chip_idx_pair[0]
        sec_chip_idx = chip_idx_pair[1]

        min_cost = np.inf
        nearest_commu_qubit = []
        commu_qubit_pair_cost = {}
        for commu_qubit_1 in each_chip_commu_qubits[fst_chip_idx]:
            for commu_qubit_2 in each_chip_commu_qubits[sec_chip_idx]:
                current_cost = cost_matrix[commu_qubit_1][commu_qubit_2]

                commu_pair = (commu_qubit_1, commu_qubit_2)
                commu_qubit_pair_cost[commu_pair] = current_cost

                if current_cost < min_cost:
                    min_cost = current_cost

        for commu_pair, cost in commu_qubit_pair_cost.items():
            if cost == min_cost:
                nearest_commu_qubit.append(commu_pair)
        chips_nearest_commu_qubit[chip_idx_pair] = nearest_commu_qubit

    return chips_nearest_commu_qubit
