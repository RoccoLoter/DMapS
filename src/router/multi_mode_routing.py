import time
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from bidict import bidict
from copy import deepcopy
from typing import Dict, Tuple
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.circuit import QuantumRegister, Qubit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag, dag_to_circuit

from router.car_layout_routing import CARSwap
from mapper.two_phase_mapper import TwoPhaseMapper
from router.multi_mode_layout_swap import MultiModeSwap
from frontend.chips_info_reader import QuHardwareInfoReader, ChipsNet
from frontend.create_matrix import (
    create_cost_matrix,
    create_dist_matrix,
    create_chip_dist_matrix,
)
from comparison_algorithms.mhsa.mhsa_mapper import MHSAMapper
from comparison_algorithms.pytket_dqc.init_pytket_dqc_map import tket_dqc_map
from comparison_algorithms.pytket_dqc.tket_dqc_rout import tketdqc_local_rout

logger = logging.getLogger(__name__)


class MultiModeRouting:
    def __init__(self) -> None:
        self.max_iteration = 5
        self.mapping_time = 0
        self.routing_time = 0

        self.qc_list = []
        self.oqc_list = []

    def mhsa_map_sabre_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_cost_matrix: np.ndarray = None,
        chip_type="zcz",
        heuristic="lookahead",
        lookahead_ability: int = 20,
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on MHSA algorithm.
        And using the sabre algorithm to routing the quantum circuit.
        """
        # The information of quantum chip network
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, chip_network = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type
        )
        pqubits_degree = chip_network.obtain_qubit_degree()

        cost_matrix = None
        if input_cost_matrix is None:
            cost_matrix = init_qubit_dist_matrix_v2(chip_network)
        else:
            cost_matrix = input_cost_matrix

        cost_matrix_tolist = cost_matrix.tolist()
        chip_dist_matrix = init_chip_dist_matrix(chip_network)

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = chip_network.chips
            chip_connections = chip_network.chip_connections
            num_phy_qubits = len(chip_network.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)
        else:
            raise TranspilerError("The MHSA algorithm works on quantum chip network.")
        coupling_map = CouplingMap(coupling_info)  # Create a instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            each_chip_qubits = chip_network.get_each_chip_qubits_idx()
            init_mapper = MHSAMapper()
            init_mapping_res = init_mapper.run(
                qu_circuit, each_chip_qubits, pqubits_degree, chip_dist_matrix, chip_network
            )
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        rout_start = time.time()
        # The process of sabre swap
        router = MultiModeSwap(
            coupling_map=coupling_map,
            heuristic=heuristic,
            exe_mode="sabre",
            input_layout=init_layout,
            chips_net=chip_network,
            input_matrix_tolist=cost_matrix_tolist,
            lookahead_ability=lookahead_ability,
        )
        final_dag = router.run(qu_dag)
        final_circuit = dag_to_circuit(final_dag)
        tmp_final_map_res = router.property_set["final_layout"]._v2p
        final_map_res = {k.index : v for k, v in tmp_final_map_res.items()}

        self.qc_list = router.qc_list
        self.oqc_list = router.oqc_list

        rout_end = time.time()
        self.mapping_time = map_end - map_start
        self.routing_time = rout_end - rout_start

        if chip_type == "zcz":
            qubits_name_n_idx = chip_network.qubits_n_index

            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)
            new_final_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

            new_final_circuit = self._conv_circ_qubits_idx(
                final_circuit, qubits_name_n_idx
            )

            return new_init_mapping, new_final_mapping, new_final_circuit
            # return tmp_init_mapping, new_final_mapping, new_final_circuit # code_verify

        # @param tmp_data(dict): The input initial mapping result.
        # @param final_mapping_res(dict): The final mapping result.
        # @param final_circuit(QuantumCircuit): The mapped quantum circuit.
        return tmp_init_mapping, final_map_res, final_circuit

    def mhsa_map_car_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_dist_matrix: np.ndarray = None,
        chip_type: str = "zcz",
        heuristic: str = "lookahead",
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on MHSA algorithm.
        And using the car algorithm to routing the quantum circuit.
        """
        # The information of quantum chip hardware
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, chip_network = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type
        )

        # The degree information of physical qubits
        pqubits_degree = chip_network.obtain_qubit_degree()

        dist_matrix = None
        if input_dist_matrix is None:
            dist_matrix = init_qubit_dist_matrix_v2(chip_network)
        else:
            dist_matrix = input_dist_matrix
        chip_dist_matrix = init_chip_dist_matrix(chip_network)
        qubit_dist_matrix_tolist = dist_matrix.tolist()
        chip_dist_matrix_tolist = chip_dist_matrix.tolist()

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = chip_network.chips
            chip_connections = chip_network.chip_connections
            num_phy_qubits = len(chip_network.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)
        else:
            raise TranspilerError("The MHSA algorithm works on quantum chip network.")
        coupling_map = CouplingMap(coupling_info)  # Create a instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            each_chip_qubits = chip_network.get_each_chip_qubits_idx()
            init_mapper = MHSAMapper()
            init_mapping_res = init_mapper.run(
                qu_circuit, each_chip_qubits, pqubits_degree, chip_dist_matrix, chip_network
            )
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        rout_start = time.time()
        # The process of routing.
        router = CARSwap(
            coupling_map=coupling_map,
            chips_net=chip_network,
            cost_matrix_tolist=qubit_dist_matrix_tolist,
            chip_dist_matrix_tolist=chip_dist_matrix_tolist,
            heuristic=heuristic,
            fake_run=False,
            input_layout=init_layout,
        )
        final_dag = router.run(qu_dag)
        final_circuit = dag_to_circuit(final_dag)
        tmp_final_map_res = router.property_set["final_layout"]._v2p
        final_map_res = {k.index : v for k, v in tmp_final_map_res.items()}
        
        self.qc_list = router.qc_list
        self.oqc_list = router.oqc_list

        rout_end = time.time()
        self.mapping_end = map_end - map_start
        self.routing_time = rout_end - rout_start

        if chip_type == "zcz":
            qubits_name_n_idx = chip_network.qubits_n_index

            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)
            new_final_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

            new_final_circuit = self._conv_circ_qubits_idx(
                final_circuit, qubits_name_n_idx
            )

            # @param new_init_mapping(dict): The input initial mapping result.
            # @param new_final_mapping(dict): The final mapping result.
            # @param new_final_circuit(QuantumCircuit): The mapped quantum circuit.
            return new_init_mapping, new_final_mapping, new_final_circuit
            # return tmp_init_mapping, new_final_mapping, new_final_circuit # code_verify

        # @param tmp_data(dict): The input initial mapping result.
        # @param tmp_data(dict): The final mapping result.
        # @param final_circuit(QuantumCircuit): The mapped quantum circuit.
        return tmp_init_mapping, final_map_res, final_circuit

    def cpa_map_sabre_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_dist_matrix: np.ndarray = None,
        input_cost_matrix: np.ndarray = None,
        chip_type="zcz",
        heuristic="lookahead",
        is_global_opt: bool = True,
        lookahead_ability: int = 20,
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on hypergraph partitioning and hyper_heuristic sub- circuits assignment (named: CPA).
        And using the sabre algorithm to routing the quantum circuit.
        """
        # The information of quantum chip network
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, qu_hardware_info = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type
        )
        qubit_dist_matrix = None
        cost_matrix = None
        if input_dist_matrix is None:
            qubit_dist_matrix = init_qubit_dist_matrix_v2(qu_hardware_info)
        else:
            qubit_dist_matrix = input_dist_matrix
        if input_cost_matrix is None:
            cost_matrix = init_cost_matrix(qu_hardware_info)
        else:
            cost_matrix = input_cost_matrix

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = qu_hardware_info.chips
            chip_connections = qu_hardware_info.chip_connections
            num_phy_qubits = len(qu_hardware_info.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)
        else:
            raise TranspilerError(
                "The CPA compilation algorithm works on quantum chip network."
            )
        coupling_map = CouplingMap(coupling_info)  # Create a instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            init_mapper = TwoPhaseMapper(fst_phase_iter=50, sec_phase_iter=200)

            if is_global_opt:
                init_mapping_res = init_mapper.run(
                    quantum_circuit=qu_circuit,
                    config_fp=chip_info_fn,
                    chip_type=chip_type,
                    dist_matrix=qubit_dist_matrix,
                    is_global_calculate=True,
                )
            else:
                init_mapping_res = init_mapper.run(
                    quantum_circuit=qu_circuit,
                    config_fp=chip_info_fn,
                    chip_type=chip_type,
                    dist_matrix=qubit_dist_matrix,
                    is_global_calculate=False,
                )
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        rout_start = time.time()
        cost_matrix_tolist = cost_matrix.tolist()
        # The process of sabre swap
        router = MultiModeSwap(
            coupling_map=coupling_map,
            heuristic=heuristic,
            exe_mode="sabre",
            input_layout=init_layout,
            chips_net=qu_hardware_info,
            input_matrix_tolist=cost_matrix_tolist,
            lookahead_ability=lookahead_ability,
        )
        final_dag = router.run(qu_dag)
        final_circuit = dag_to_circuit(final_dag)
        tmp_final_map_res = router.property_set["final_layout"]._v2p
        final_map_res = {k.index : v for k, v in tmp_final_map_res.items()}

        self.qc_list = router.qc_list
        self.oqc_list = router.oqc_list

        rout_end = time.time()
        self.mapping_time = map_end - map_start
        self.routing_time = rout_end - rout_start

        if chip_type == "zcz":
            qubits_name_n_idx = qu_hardware_info.qubits_n_index

            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)
            new_final_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

            new_final_circuit = self._conv_circ_qubits_idx(
                final_circuit, qubits_name_n_idx
            )

            return new_init_mapping, new_final_mapping, new_final_circuit
            # return tmp_init_mapping, new_final_mapping, new_final_circuit # code_verify

        # @param tmp_data(dict): The input initial mapping result.
        # @param final_mapping_res(dict): The final mapping result.
        # @param final_circuit(QuantumCircuit): The mapped quantum circuit.
        return tmp_init_mapping, final_map_res, final_circuit

    def cpa_map_car_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_dist_matrix: np.ndarray = None,
        input_chip_dist_matrix: np.ndarray = None,
        chip_type: str = "zcz",
        heuristic: str = "lookahead",
        intra_chip_all2all: bool = False,
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on hypergraph partitioning and hyper_heuristic sub- circuits assignment (named: CPA).
        And using the car algorithm to routing the quantum circuit.
        """
        # The information of quantum chip hardware
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, qu_hardware_info = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type, intra_chip_all2all=intra_chip_all2all
        )
        dist_matrix = None
        chip_dist_matrix = None
        if input_dist_matrix is None:
            dist_matrix = init_qubit_dist_matrix_v2(qu_hardware_info)
        else:
            dist_matrix = input_dist_matrix
        if input_chip_dist_matrix is None:
            chip_dist_matrix = init_chip_dist_matrix(qu_hardware_info)
        else:
            chip_dist_matrix = input_chip_dist_matrix
        dist_matrix_tolist = dist_matrix.tolist()
        chip_dist_matrix_tolist = chip_dist_matrix.tolist()

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = qu_hardware_info.chips
            chip_connections = qu_hardware_info.chip_connections
            num_phy_qubits = len(qu_hardware_info.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)

        else:
            raise TranspilerError(
                "The qubit mapping algorithm works on quantum chip network."
            )
        coupling_map = CouplingMap(coupling_info)  # Create an instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            init_mapper = TwoPhaseMapper(fst_phase_iter=100, sec_phase_iter=200)
            init_mapping_res = init_mapper.run(
                quantum_circuit=qu_circuit,
                config_fp=chip_info_fn,
                chip_type=chip_type,
                dist_matrix=dist_matrix,
                intra_chip_all2all=intra_chip_all2all,
            )
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        if chip_type == "zcz":
            qubits_name_n_idx = qu_hardware_info.qubits_n_index
            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

        rout_start = time.time()
        # The process of routing.
        router = CARSwap(
            coupling_map=coupling_map,
            chips_net=qu_hardware_info,
            cost_matrix_tolist=dist_matrix_tolist,
            chip_dist_matrix_tolist=chip_dist_matrix_tolist,
            heuristic=heuristic,
            fake_run=False,
            input_layout=init_layout,
            intra_chip_all2all=intra_chip_all2all,
        )
        final_dag = router.run(qu_dag)
        final_circuit = dag_to_circuit(final_dag)
        tmp_final_map_res = router.property_set["final_layout"]._v2p
        final_map_res = {k.index : v for k, v in tmp_final_map_res.items()}

        self.qc_list = router.qc_list
        self.oqc_list = router.oqc_list

        rout_end = time.time()
        self.mapping_time = map_end - map_start
        self.routing_time = rout_end - rout_start

        if chip_type == "zcz":
            qubits_name_n_idx = qu_hardware_info.qubits_n_index

            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)
            new_final_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

            new_final_circuit = self._conv_circ_qubits_idx(
                final_circuit, qubits_name_n_idx
            )

            # @param new_init_mapping(dict): The input initial mapping result.
            # @param new_final_mapping(dict): The final mapping result.
            # @param new_final_circuit(QuantumCircuit): The mapped quantum circuit.
            return new_init_mapping, new_final_mapping, new_final_circuit
            # return tmp_init_mapping, new_final_mapping, new_final_circuit # code_verify

        # @param tmp_data(dict): The input initial mapping result.
        # @param tmp_data(dict): The final mapping result.
        # @param final_circuit(QuantumCircuit): The mapped quantum circuit.
        return tmp_init_mapping, final_map_res, final_circuit

    def tketdqc_map_sabre_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_cost_matrix: np.ndarray = None,
        input_qubit_dist_matrix: np.ndarray = None,
        input_chip_dist_matrix: np.ndarray = None,
        chip_type="zcz",
        heuristic="lookahead",
        lookahead_ability:int =20,
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on MHSA algorithm.
        And using the sabre algorithm to routing the quantum circuit.
        """
        # The information of quantum chip network
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, chip_network = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type
        )

        cost_matrix = None
        qubit_dist_matrix = None
        chip_dist_matrix = None
        if input_cost_matrix is None:
            cost_matrix = init_cost_matrix(chip_network)
        else:
            cost_matrix = input_cost_matrix
        if input_qubit_dist_matrix is None:
            qubit_dist_matrix = init_qubit_dist_matrix_v2(chip_network)
        else:
            qubit_dist_matrix = input_qubit_dist_matrix
        if input_chip_dist_matrix is None:
            chip_dist_matrix = init_chip_dist_matrix(chip_network)
        else:
            chip_dist_matrix = input_chip_dist_matrix

        cost_matrix_tolist = cost_matrix.tolist()

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = chip_network.chips
            chip_connections = chip_network.chip_connections
            num_phy_qubits = len(chip_network.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)
        else:
            raise TranspilerError("The MHSA algorithm works on quantum chip network.")
        coupling_map = CouplingMap(coupling_info)  # Create a instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            _, _, init_mapping_res = tket_dqc_map(qu_circuit, chip_network)
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        rout_start = time.time()
        # The process of sabre swap
        router = MultiModeSwap(
            coupling_map=coupling_map,
            heuristic=heuristic,
            exe_mode="sabre",
            input_layout=init_layout,
            chips_net=chip_network,
            input_matrix_tolist=cost_matrix_tolist,
            lookahead_ability=lookahead_ability,
        )
        final_dag = router.run(qu_dag)
        final_circuit = dag_to_circuit(final_dag)
        tmp_final_map_res = router.property_set["final_layout"]._v2p
        final_map_res = {k.index : v for k, v in tmp_final_map_res.items()}

        self.qc_list = router.qc_list
        self.oqc_list = router.oqc_list

        rout_end = time.time()
        self.mapping_time = map_end - map_start
        self.routing_time = rout_end - rout_start

        if chip_type == "zcz":
            qubits_name_n_idx = chip_network.qubits_n_index

            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)
            new_final_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

            new_final_circuit = self._conv_circ_qubits_idx(
                final_circuit, qubits_name_n_idx
            )

            return new_init_mapping, new_final_mapping, new_final_circuit
            # return tmp_init_mapping, new_final_mapping, new_final_circuit # code_verify

        # @param tmp_data(dict): The input initial mapping result.
        # @param final_mapping_res(dict): The final mapping result.
        # @param final_circuit(QuantumCircuit): The mapped quantum circuit.
        return tmp_init_mapping, final_map_res, final_circuit

    def tketdqc_map_car_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_qubit_dist_matrix: np.ndarray = None,
        input_chip_dist_matrix: np.ndarray = None,
        chip_type="zcz",
        heuristic="lookahead",
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on MHSA algorithm.
        And using the sabre algorithm to routing the quantum circuit.
        """
        # The information of quantum chip network
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, chip_network = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type
        )

        qubit_dist_matrix = None
        chip_dist_matrix = None
        if input_qubit_dist_matrix is None:
            qubit_dist_matrix = init_qubit_dist_matrix_v2(chip_network)
        else:
            qubit_dist_matrix = input_qubit_dist_matrix
        if input_chip_dist_matrix is None:
            chip_dist_matrix = init_chip_dist_matrix(chip_network)
        else:
            chip_dist_matrix = input_chip_dist_matrix

        qubit_dist_matrix_tolist = qubit_dist_matrix.tolist()
        chip_dist_matrix_tolist = chip_dist_matrix.tolist()

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = chip_network.chips
            chip_connections = chip_network.chip_connections
            num_phy_qubits = len(chip_network.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)
        else:
            raise TranspilerError("The MHSA algorithm works on quantum chip network.")
        coupling_map = CouplingMap(coupling_info)  # Create a instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            _, _, init_mapping_res = tket_dqc_map(qu_circuit, chip_network)
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        rout_start = time.time()
        # The process of routing.
        router = CARSwap(
            coupling_map=coupling_map,
            chips_net=chip_network,
            cost_matrix_tolist=qubit_dist_matrix_tolist,
            chip_dist_matrix_tolist=chip_dist_matrix_tolist,
            heuristic=heuristic,
            fake_run=False,
            input_layout=init_layout,
        )
        final_dag = router.run(qu_dag)
        final_circuit = dag_to_circuit(final_dag)
        tmp_final_map_res = router.property_set["final_layout"]._v2p
        final_map_res = {k.index : v for k, v in tmp_final_map_res.items()}

        self.qc_list = router.qc_list
        self.oqc_list = router.oqc_list

        rout_end = time.time()
        self.mapping_end = map_end - map_start
        self.routing_time = rout_end - rout_start

        if chip_type == "zcz":
            qubits_name_n_idx = chip_network.qubits_n_index

            new_init_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)
            new_final_mapping = self._conv_mapping_res(tmp_data, qubits_name_n_idx)

            new_final_circuit = self._conv_circ_qubits_idx(
                final_circuit, qubits_name_n_idx
            )

            # @param new_init_mapping(dict): The input initial mapping result.
            # @param new_final_mapping(dict): The final mapping result.
            # @param new_final_circuit(QuantumCircuit): The mapped quantum circuit.
            return new_init_mapping, new_final_mapping, new_final_circuit
            # return tmp_init_mapping, new_final_mapping, new_final_circuit # code_verify

        # @param tmp_data(dict): The input initial mapping result.
        # @param tmp_data(dict): The final mapping result.
        # @param final_circuit(QuantumCircuit): The mapped quantum circuit.
        return tmp_init_mapping, final_map_res, final_circuit

    def tketdqc_map_tketdqc_rout(
        self,
        origin_qasm_fn: Path,
        chip_info_fn: Path,
        mapping_res: Dict[Qubit, int] = None,
        input_cost_matrix: np.ndarray = None,
        input_qubit_dist_matrix: np.ndarray = None,
        input_chip_dist_matrix: np.ndarray = None,
        chip_type="zcz",
        heuristic="lookahead",
        lookahead_ability:int =20,
        intra_chip_all2all: bool = False,
    ) -> Tuple[Dict[str, int], Dict[int, int], QuantumCircuit]:
        """
        Obtain the qubits initial mapping based on MHSA algorithm.
        And using the sabre algorithm to routing the quantum circuit.
        """
        # The information of quantum chip network
        qu_hardware_obj = QuHardwareInfoReader(chip_info_fn)
        is_has_multi_chips, chip_network = qu_hardware_obj.get_hardware_info(
            chip_type=chip_type, intra_chip_all2all=intra_chip_all2all
        )

        cost_matrix = None
        qubit_dist_matrix = None
        chip_dist_matrix = None
        if input_cost_matrix is None:
            cost_matrix = init_cost_matrix(chip_network)
        else:
            cost_matrix = input_cost_matrix
        if input_qubit_dist_matrix is None:
            qubit_dist_matrix = init_qubit_dist_matrix(chip_network)
        else:
            qubit_dist_matrix = input_qubit_dist_matrix
        if input_chip_dist_matrix is None:
            chip_dist_matrix = init_chip_dist_matrix(chip_network)
        else:
            chip_dist_matrix = input_chip_dist_matrix

        cost_matrix_tolist = cost_matrix.tolist()

        num_phy_qubits = None  # The number of physical qubits.
        coupling_info = []

        if is_has_multi_chips:
            # Before activating the flow of mapping and scheduling,
            # we should get the chip information.
            # Here, we only need the two qubits coupling information
            chips = chip_network.chips
            chip_connections = chip_network.chip_connections
            num_phy_qubits = len(chip_network.get_total_qubits())

            for chip in chips:
                couplings = chip.couplings
                for coupling in couplings:
                    phy_qubit_idx_pair = [coupling[0], coupling[1]]
                    coupling_info.append(phy_qubit_idx_pair)

            # Add the information of remote physical connection.
            for chip_connection in chip_connections:
                qubit_pair = chip_connection.qubit_pair
                qubit_idx_pair = [qubit_pair[0].index, qubit_pair[1].index]
                coupling_info.append(qubit_idx_pair)
        else:
            raise TranspilerError("The MHSA algorithm works on quantum chip network.")
        coupling_map = CouplingMap(coupling_info)  # Create a instance of CouplingMap.

        # Create a new quantum circuit and tag
        qu_circuit = QuantumCircuit.from_qasm_file(origin_qasm_fn)
        if qu_circuit.num_qubits > num_phy_qubits:
            raise TranspilerError(
                "Error: The number of qubits in the quantum circuit is greater than the number of physical qubits!"
            )
        _, new_qu_circuit = generate_new_circuit(qu_circuit, num_phy_qubits)
        qu_dag = circuit_to_dag(new_qu_circuit)

        # Get the initial qubits mapping result.
        init_mapping_res = None
        map_start = time.time()
        if mapping_res is None:
            _, qubit_group_res, init_mapping_res, num_eprs = tket_dqc_map(qu_circuit, chip_network, "anneal", "distribution")
        else:
            init_mapping_res = mapping_res
        map_end = time.time()

        # Remove remote two-qubit gates from quantum circuits
        vqubits_n_chips = {vqubit_idx: chip_idx for chip_idx, vqubits in qubit_group_res.items() for vqubit_idx in vqubits}
        remove_2q_circuit = QuantumCircuit(name="remove_2q_circuit", global_phase=new_qu_circuit.global_phase)
        for qreg in new_qu_circuit.qregs:
            remove_2q_circuit.add_register(qreg)
        for creg in new_qu_circuit.cregs:
            remove_2q_circuit.add_register(creg)
        for insn in new_qu_circuit:
            instruction = insn[0]
            if instruction.name == "cx" or instruction.name == "cz":
                fst_vqubit, snd_vqubit = insn[1][0], insn[1][1]
                if vqubits_n_chips[fst_vqubit.index] != vqubits_n_chips[snd_vqubit.index]:
                    continue
                qubits = [qubit for qubit in insn[1]]
                clbits = [clbit for clbit in insn[2]]
                remove_2q_circuit.append(instruction, qubits, clbits)
        remove_2q_dag = circuit_to_dag(remove_2q_circuit)

        #! Only consider the single dag register.
        """
        - Process the obtained qubits initial mapping result.
        - The initial mapping result before processing is a dictionary, the key is the qubit(Qubit),and the value is the index of physical qubit(int).
        - After processing, the initial mapping result also is a dictionary, the key is the index of virtual qubit index(int), and the value is the index of physical qubit(int).
        - Based on the newly generated initial mapping result, a new layout(Layout) is generated.
        """
        init_layout = Layout()
        tmp_data = {}
        for k, v in init_mapping_res.items():
            tmp_data[k.index] = v
        tmp_init_mapping = create_tmp_initial_mapping(tmp_data, num_phy_qubits)
        dag_registers = qu_dag.qregs
        # Create the input dictionary data for the initial layout.
        if len(dag_registers) == 1:
            canonical_register = None
            for _, reg in dag_registers.items():
                canonical_register = reg

            input_dict = {}
            for k, v in tmp_init_mapping.items():
                input_dict[Qubit(register=canonical_register, index=k)] = v
            init_layout.from_dict(input_dict=input_dict)
        else:
            raise TranspilerError("To be realized.")

        rout_start = time.time()
        # The process of sabre swap
        router = tketdqc_local_rout(
            coupling_map=coupling_map,
            chips_net=chip_network,
            heuristic=heuristic,
            exe_mode="sabre",
            input_layout=init_layout,
            input_matrix_tolist=cost_matrix_tolist,
            lookahead_ability=lookahead_ability,
        )
        final_dag = router.run(remove_2q_dag)
        final_circuit = dag_to_circuit(final_dag)

        rout_end = time.time()

        num_swaps = 0
        for insn in final_circuit:
            if insn[0].name == "swap":
                num_swaps += 1
        self.mapping_time = map_end - map_start
        self.routing_time = rout_end - rout_start

        return tmp_data, tmp_data, final_circuit, num_eprs, num_swaps

    def _analyze_chip_candidates(
        self, num_virtual_qubits: int, chip_scale_info: Dict[int, int]
    ) -> Dict[int, Dict[int, int]]:
        """
        Analyze all the chip cluster candidates.
        """
        chip_clus_candidates = {}

        # Get the all subset of the chip index list.
        all_subsets = []
        chips_idx = list(chip_scale_info.keys())
        num_chips = len(chips_idx)
        for num in range(num_chips):
            for subset in combinations(chips_idx, num + 1):
                all_subsets.append(subset)

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

        return chip_clus_candidates

    def _is_usable_partiton(
        self, par_res: Dict[Qubit, int], chips_scale_info: Dict[int, int]
    ):
        """
        Judge if the partition result is usable.
        """
        is_usable = True

        # Count the size of each partitioned block.
        blocks_info = {}
        for _, block_id in par_res.items():
            if block_id not in blocks_info:
                blocks_info[block_id] = 0
            else:
                blocks_info[block_id] += 1

        # Judge if the size of each partitioned block is less than the size of the chip.
        sorted_chips_scale = sorted(
            chips_scale_info.items(), key=lambda x: x[1], reverse=True
        )
        sorted_blocks_size = sorted(
            blocks_info.items(), key=lambda x: x[1], reverse=True
        )

        for i in range(len(blocks_info)):
            if sorted_blocks_size[i][1] > sorted_chips_scale[i][1]:
                is_usable = False
                break

        return is_usable

    def _conv_mapping_res(
        self, ori_mapping_res: Dict, qubits_name_n_idx: bidict
    ) -> Dict:
        """Convert the physical qubits in the qubits mapping result."""
        new_mapping_res = {}

        for k, v in ori_mapping_res.items():
            qubit_name = qubits_name_n_idx.inverse[v]
            new_qubit_idx = int(qubit_name[1:])
            new_mapping_res[k] = new_qubit_idx

        return new_mapping_res

    def _conv_circ_qubits_idx(
        self, qu_circuit: QuantumCircuit, qubits_name_n_idx: bidict
    ) -> QuantumCircuit:
        """Convert the subscript of qubits in the mapped quantum circuit."""
        new_circ = QuantumCircuit(
            name=qu_circuit.name, global_phase=qu_circuit.global_phase
        )

        # Add the information of quantum registers.
        #! The default value here is 66 for the time being, and further optimization will be carried out later.
        canonical_register = QuantumRegister(name="q", size=66)
        new_circ.add_register(canonical_register)

        # Add the information of classical registers.
        for creg in qu_circuit.cregs:
            new_circ.add_register(creg)

        # Add the information of operations.
        for insn in qu_circuit:
            instruction = insn[0]

            new_qubits = []
            for qubit in insn[1]:
                qubit_name = qubits_name_n_idx.inverse[qubit.index]
                new_qubit_idx = int(qubit_name[1:])
                new_qubits.append(
                    Qubit(register=canonical_register, index=new_qubit_idx)
                )

            clbits = [clbit for clbit in insn[2]]
            new_circ.append(instruction, new_qubits, clbits)
        return new_circ

    def _conv_circ(
        self,
        qu_circuit: QuantumCircuit,
        mapping_res: bidict,
        num_phy_qubits: int,
    ) -> QuantumCircuit:
        """Convert the subscript of qubits in the mapped quantum circuit."""
        new_circ = QuantumCircuit(
            name=qu_circuit.name, global_phase=qu_circuit.global_phase
        )

        # Add the information of quantum registers.
        canonical_register = QuantumRegister(name="q", size=num_phy_qubits)
        new_circ.add_register(canonical_register)

        # Add the information of classical registers.
        for creg in qu_circuit.cregs:
            new_circ.add_register(creg)

        # Add the information of operations.
        for insn in qu_circuit:
            instruction = insn[0]

            new_qubits = []
            for qubit in insn[1]:
                vir_qubit_idx = qubit.index
                phy_qubit_idx = mapping_res[vir_qubit_idx]
                new_qubits.append(
                    Qubit(register=canonical_register, index=phy_qubit_idx)
                )

            clbits = [clbit for clbit in insn[2]]
            new_circ.append(instruction, new_qubits, clbits)
        return new_circ


# TODO
# 1. How to interact with the quantum circuit after obtaining the initial mapping?
# 2. How to handle the situation of multiple qreg?
def generate_new_circuit(input_circ: QuantumCircuit, size: int):
    """
    Generate a new circuit which the size of quantum register is equal to the number of physical qubits.
    """
    circ = QuantumCircuit(name=input_circ.name, global_phase=input_circ.global_phase)

    qubits_info = {}
    qubit_idx = 0
    for qubit in input_circ.qubits:
        qubits_info[qubit.register.name + str(qubit.index)] = qubit_idx
        qubit_idx += 1

    canonical_register = QuantumRegister(name="q", size=size)
    circ.add_register(canonical_register)

    # for creg in input_circ.cregs:
    #     new_creg = ClassicalRegister(name=creg.name, size=size)
    #     new_cregs_dict[creg.name] = new_creg
    #     circ.add_register(new_creg)
    for creg in input_circ.cregs:
        circ.add_register(creg)

    for insn in input_circ:
        instruction = insn[0]
        qubits = [
            Qubit(register=canonical_register, index=qubit.index) for qubit in insn[1]
        ]
        clbits = [clbit for clbit in insn[2]]
        circ.append(instruction, qubits, clbits)
    return qubits_info, circ


def create_tmp_initial_mapping(input_mapping, qubits_num) -> bidict:
    """Create a new temporary qubits mapping relationship."""
    final_mapping_result = {}

    key_list = list(input_mapping.keys())  # The key list of the initial mapping
    if isinstance(key_list[0], Qubit):
        canonical_reg = QuantumRegister(name=key_list[0].register.name, size=qubits_num)
        value_list = list(input_mapping.values())

        idx_list_for_key = [i for i in range(qubits_num)]
        idx_list_for_value = deepcopy(idx_list_for_key)

        # Gets the index that was not matched
        for key in key_list:
            idx_list_for_key.remove(key.index)
        for value in value_list:
            idx_list_for_value.remove(value)

        for k, v in input_mapping.items():
            qubit = Qubit(register=canonical_reg, index=k.index)
            final_mapping_result[qubit] = v
        for i in range(len(idx_list_for_key)):
            fake_qubit = Qubit(register=canonical_reg, index=idx_list_for_value[i])
            final_mapping_result[fake_qubit] = idx_list_for_value[i]
    else:
        value_list = list(
            input_mapping.values()
        )  # The value list of the initial mapping

        idx_list_for_key = [i for i in range(qubits_num)]
        idx_list_for_value = deepcopy(idx_list_for_key)

        # Gets the index that was not matched
        for key in key_list:
            idx_list_for_key.remove(key)
        for value in value_list:
            idx_list_for_value.remove(value)

        for k, v in input_mapping.items():
            final_mapping_result[k] = v
        for i in range(len(idx_list_for_key)):
            final_mapping_result[idx_list_for_key[i]] = idx_list_for_value[i]

    return final_mapping_result


def init_cost_matrix(chip_network: ChipsNet, remote_dist=None) -> np.ndarray:
    commu_qubit_n_chip_idx = {}

    total_qubits_topology = chip_network.obtain_total_chip_network()
    each_chip_commu_qubits = chip_network.get_each_chip_commu_qubits_idx()
    for chip_idx, idx_list in each_chip_commu_qubits.items():
        for qubit_idx in idx_list:
            commu_qubit_n_chip_idx[qubit_idx] = chip_idx

    cost_matrix = None
    if remote_dist is None:
        cost_matrix = create_cost_matrix(total_qubits_topology, commu_qubit_n_chip_idx)
    else:
        cost_matrix = create_cost_matrix(
            total_qubits_topology, commu_qubit_n_chip_idx, remote_dist=remote_dist
        )

    return cost_matrix


def init_chip_dist_matrix(chip_network: ChipsNet) -> np.ndarray:
    chips_topology = chip_network.obtain_chip_network()
    chip_dist_matrix = create_chip_dist_matrix(chips_topology)

    return chip_dist_matrix


def init_qubit_dist_matrix(chip_network: ChipsNet) -> np.ndarray:
    total_qubits_topology = chip_network.obtain_total_chip_network()
    qubit_dist_matrix = create_dist_matrix(total_qubits_topology)

    return qubit_dist_matrix


def init_qubit_dist_matrix_v2(chip_network: ChipsNet) -> np.ndarray:
    total_qubits_topology = chip_network.obtain_total_chip_network()
    qubits_idx = list(total_qubits_topology.nodes())
    tmp_dist_matrix = nx.floyd_warshall_numpy(total_qubits_topology, qubits_idx)
    new_dist_matrix = np.zeros((len(qubits_idx), len(qubits_idx)))
    for i in range(len(qubits_idx)):
        for j in range(len(qubits_idx)):
            new_dist_matrix[qubits_idx[i]][qubits_idx[j]] = tmp_dist_matrix[i][j]
            new_dist_matrix[qubits_idx[j]][qubits_idx[i]] = tmp_dist_matrix[j][i]

    return new_dist_matrix
