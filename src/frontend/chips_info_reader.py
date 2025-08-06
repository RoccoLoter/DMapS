import json
import itertools
import networkx as nx
from pathlib import Path
from bidict import bidict
from networkx import Graph, MultiGraph
from typing import List, Set, Dict, Tuple, Any

from global_config import repo_path
from frontend.parser_calibration.parser_calibration_data import (
    parser_ibm_calib_data,
    parser_zcz_calib_data,
)

REMOTE_CONNECTION_WEIGHT = 10


class ScQubit:
    def __init__(
        self,
        qubit_idx: int,
        chip_idx: int,
        is_comm_qubit: bool,
        single_qubit_gate_fid: Dict[str, float],
    ) -> None:
        """
        Args:
            qubit_idx           : The global index of physical qubit in the quantum chip network.
            chip_idx            : The index of quantum chip to which the physical qubit belongs.
            is_comm_qubit       : This is a bool value, it indicates whether physical qubit is used for remote connections.
            single_qubit_gate_fid: The fidelity information of each single gate that acts on this physical qubit.
        """
        self._global_idx = qubit_idx
        self._parent = chip_idx
        self._is_comm_q = is_comm_qubit
        self._single_qubit_gate_fid = single_qubit_gate_fid
        self._qubit_name = None

    @property
    def index(self) -> int:
        """Get the global index of physical qubit."""
        return self._global_idx

    @property
    def is_comm_qubit(self) -> bool:
        """Show whether the physical qubit is communication qubit, and returns True if so."""
        return self._is_comm_q

    @property
    def name(self) -> str:
        """Get the name of physical qubit."""
        return self._qubit_name

    def get_rela_single_gate_fid(self) -> dict:
        """Get the fidelity information of each single gate that acts on this physical qubit."""
        return self._single_qubit_gate_fid

    def get_belong_chip_idx(self) -> int:
        """Get the serial number of the quantum chip where the physical qubit is located."""
        return self._parent

    def set_qubit_name(self, name: str) -> None:
        """Set the name of physical qubit."""
        if name != "":
            self._qubit_name = name
        else:
            raise ValueError("The name of physical qubit cannot be empty!")


class Connection:
    def __init__(self, qubit0: ScQubit, qubit1: ScQubit, fidelity: float) -> None:
        """
        Args:
            qubit0(ScQubit): The first physical qubit in the connection.
            qubit1(ScQubit): The second physical qubit in the connection.
            fidelity : The fidelity of the connection between the two qubits.
        """
        self._qubit0 = qubit0
        self._qubit1 = qubit1
        self._fidelity = fidelity

    @property
    def qubit_pair(self) -> Tuple[ScQubit, ScQubit]:
        """Get the qubit pair of connection."""
        return (self.qubit0, self.qubit1)

    @property
    def qubit0(self) -> ScQubit:
        """Get the first physical qubit in the connection."""
        return self._qubit0

    @property
    def qubit1(self) -> ScQubit:
        """Get the second physical qubit in the connection."""
        return self._qubit1

    @property
    def fidelity(self) -> float:
        """Get the fidelity of connection."""
        return self._fidelity


class ScChip:
    def __init__(self, chip_name: str, chip_idx: int) -> None:
        """
        Args:
            chip_name         : The name of quantum chip.
            chip_idx          : The serial number of quantum chip.
        """
        # About chip informtaion
        self._chip_name = chip_name
        self._chip_idx = chip_idx

        self._chip_topology = Graph()
        self._qubit_list = list()
        self._qubit_connections = list()

    @property
    def name(self) -> str:
        """Get the name of quantum chip."""
        return self._chip_name

    @property
    def index(self) -> int:
        """Get the index of quantum chip."""
        return self._chip_idx

    @property
    def qubits(self) -> List[ScQubit]:
        """Get the physical qubit list of quantum chip."""
        return self._qubit_list

    @property
    def qubits_num(self) -> int:
        """Get the number of physical qubits."""
        return len(self._qubit_list)

    @property
    def qubits_index(self) -> List[int]:
        """Get the index list of physical qubits."""
        return [qubit.index for qubit in self._qubit_list]

    @property
    def comm_qubits(self) -> List[ScQubit]:
        """Get the communication physical qubit list of quantum chip."""
        commu_qubits = [qubit for qubit in self._qubit_list if qubit.is_comm_qubit]
        return commu_qubits

    @property
    def commu_qubits_index(self) -> List[int]:
        """Get the index list of communication physical qubits."""
        commu_qubits_index = [
            qubit.index for qubit in self._qubit_list if qubit.is_comm_qubit
        ]
        return commu_qubits_index

    @property
    def qubit_connection_average_fid(self) -> float:
        """Get the average fidelity of qubit connections."""
        connection_average_fid = 0.0
        total_connection_fid = 0.0
        for connection in self._qubit_connections:
            total_connection_fid += connection.fidelity

        connection_average_fid = total_connection_fid / len(self._qubit_connections)
        return connection_average_fid

    @property
    def diameter(self) -> int:
        diameter = nx.diameter(self._chip_topology)
        return diameter

    @property
    def couplings(self) -> List[Tuple]:
        """Get the physical couplings of quantum chip."""
        couplings = [
            (connection.qubit0.index, connection.qubit1.index, connection.fidelity)
            for connection in self._qubit_connections
        ]
        return couplings

    def obtain_qubit_neighbours(self, qubit_idx: int) -> List[ScQubit]:
        """Get the neighbours of physical qubit."""
        neighbours = list()

        neighbours_index = self._chip_topology.neighbors(qubit_idx)
        for neighbour_index in neighbours_index:
            for qubit in self._qubit_list:
                if qubit.index == neighbour_index:
                    neighbours.append(qubit)
        return neighbours

    def add_qubit(self, qubit: ScQubit):
        """Add a physical qubit to the quantum chip topology."""
        self._qubit_list.append(qubit)
        self._chip_topology.add_node(qubit.index, qubit_obj=qubit)

    def add_connection(self, connection: Connection):
        """Add a connection between two physical qubits to the quantum chip topology."""
        if not self._chip_topology.has_edge(
            connection.qubit0.index, connection.qubit1.index
        ):
            self._qubit_connections.append(connection)
            self._chip_topology.add_edge(
                connection.qubit0.index,
                connection.qubit1.index,
                fidelity=connection.fidelity,
            )
        else:
            pass

    def get_chip_topology(self) -> Graph:
        """Get the topology of quantum chip."""
        return self._chip_topology


class ChipsConnection(Connection):
    def __init__(
        self, commu_qubit0: ScQubit, commu_qubit1: ScQubit, fidelity: float
    ) -> None:
        super().__init__(commu_qubit0, commu_qubit1, fidelity)

    @property
    def chip_idx_pair(self) -> Tuple[int, int]:
        """Get the pair of index of quantum chips where the communication qubits are located."""
        _chip_idx_pair = (
            self.qubit0.get_belong_chip_idx(),
            self.qubit1.get_belong_chip_idx(),
        )
        return _chip_idx_pair


class ChipsNet:
    def __init__(
        self, chip_set: Set[ScChip], chip_connections: Set[ChipsConnection]
    ) -> None:
        """
        Args:
            chip_set         : The set of quantum chips.
            chip_connections : The set of connections between quantum chips.
        """
        self._chip_set = chip_set
        self._remote_connections = chip_connections

        self._qubits = []
        self._qubit_n_idx = bidict()
        self._chip_network = MultiGraph()
        self._total_qubits_network = Graph()  # TODO: MultiGraph
        self._weighted_network_graph = Graph()

        for chip in self._chip_set:
            chip_index = chip.index

            self._qubits = self._qubits + chip.qubits

            self._chip_network.add_node(chip_index, label=chip)
            for qubit in chip.qubits:
                self._total_qubits_network.add_node(qubit.index, qubit_obj=qubit)
                self._weighted_network_graph.add_node(
                    qubit.index, is_commu_data_qubit=qubit._is_comm_q
                )
            for coupling in chip.couplings:
                self._total_qubits_network.add_edge(
                    coupling[0],
                    coupling[1],
                    fidelity=coupling[2],
                    swap_error=pow(1 - coupling[2], 3),
                )
                self._weighted_network_graph.add_edge(
                    coupling[0], coupling[1], weight=1
                )

        for connection in self._remote_connections:
            chip_idx_pair = connection.chip_idx_pair
            chip_1_idx = chip_idx_pair[0]
            chip_2_idx = chip_idx_pair[1]

            commu_qubit_pair = connection.qubit_pair
            commu_qubit_1_idx = commu_qubit_pair[0].index
            commu_qubit_2_idx = commu_qubit_pair[1].index

            self._chip_network.add_edge(
                chip_1_idx, chip_2_idx, remote_connection=connection
            )
            self._total_qubits_network.add_edge(
                commu_qubit_1_idx,
                commu_qubit_2_idx,
                fidelity=connection.fidelity,
                swap_error=pow(1 - connection.fidelity, 3),
            )
            self._weighted_network_graph.add_edge(
                commu_qubit_1_idx, commu_qubit_2_idx, weight=REMOTE_CONNECTION_WEIGHT
            )

    @property
    def chips(self) -> Set[ScChip]:
        """Get the set of quantum chips."""
        return self._chip_set

    @property
    def chip_connections(self) -> Set[ChipsConnection]:
        """Get the set of connections between quantum chips."""
        return self._remote_connections

    @property
    def qubits_n_index(self) -> bidict:
        """Get the index information of total physical qubits."""
        return self._qubit_n_idx

    def update_qubits_index(self, qubits_n_index: bidict):
        """Update the index information of total physical qubits."""
        self._qubit_n_idx = qubits_n_index

    def get_total_qubits(self) -> List[ScQubit]:
        """Get the total physical qubits of quantum chip network."""
        return self._qubits

    def get_each_chip_qubits(self) -> Dict[int, List[ScQubit]]:
        """Get the physical qubits of each quantum chip."""
        return {chip.index: chip.qubits for chip in self._chip_set}

    def get_each_chip_comm_qubits(self) -> Dict[int, List[ScQubit]]:
        """Get the communication physical qubits of each quantum chip."""
        return {chip.index: chip.comm_qubits for chip in self._chip_set}

    def get_each_chip_capacity(self) -> Dict[int, int]:
        """Get the capacity of each quantum chip."""
        return {chip.index: chip.qubits_num for chip in self._chip_set}

    def get_each_chip_qubits_idx(self) -> Dict[int, List[int]]:
        """Get qubit index list of each quantum chip."""
        return {chip.index: chip.qubits_index for chip in self._chip_set}

    def get_each_chip_commu_qubits_idx(self) -> Dict[int, List[int]]:
        """Get communication qubit index list of each quantum chip."""
        return {chip.index: chip.commu_qubits_index for chip in self._chip_set}

    def get_each_chip_topology(self) -> Dict[int, Graph]:
        """Get qubit network of each quantum chip."""
        return {chip.index: chip.get_chip_topology() for chip in self._chip_set}

    def obtain_total_chip_network(self) -> Graph:
        """Get the total qubit network of quantum chips."""
        return self._total_qubits_network

    def obtain_chip_network(self) -> MultiGraph:
        """Get the network of quantum chips."""
        return self._chip_network

    def obtain_weighted_network_graph(self) -> Graph:
        """Get the weighted graph of quantum chip network."""
        return self._weighted_network_graph

    def obtain_chips_idx(self) -> List[int]:
        """Get the index list of quantum chips."""
        chips_idx = [chip.index for chip in self._chip_set]
        return chips_idx

    def obtain_qubit_n_chip_idx(self) -> dict:
        """Get the index of quantum chip where the physical qubit is located."""
        qubit_n_chip_idx = {}
        for chip in self._chip_set:
            for qubit_idx in chip.qubits_index:
                qubit_n_chip_idx[qubit_idx] = chip.index
        return qubit_n_chip_idx

    def obtain_qubit_degree(self) -> dict:
        """Get the degree of each physical qubit in the quantum chip network."""
        qubit_degree = dict(self._total_qubits_network.degree())
        return qubit_degree


class QuHardwareInfoReader:
    def __init__(self, config_fn: Path) -> None:
        """
        Args:
            config_fn(Path): The path of quantum chip hardware information config file.
        """
        self._config_fn = config_fn
        self._chip_type = None
        self._intra_chip_all2all = False

    def get_hardware_info(self, chip_type: str, intra_chip_all2all: bool = False) -> Tuple[bool, ChipsNet]:
        """Get the quantum hardware information from the config file."""
        self._chip_type = chip_type
        self._intra_chip_all2all = intra_chip_all2all
        hardware_info = None

        with self._config_fn.open("r") as cf:
            config_data = json.load(cf)

        is_has_multi_chips = config_data["has multiple chips"]
        if not is_has_multi_chips:
            hardware_info = self._parse_single_chip_info(config_data)
        else:
            hardware_info = self._parse_chips_info(config_data)

        return is_has_multi_chips, hardware_info

    def _parse_single_chip_info(self, config_data: Dict[str, Any]) -> ScChip:
        """
        Read the single superconducting quantum chip information from the config file.
        """
        chip_obj = None
        is_use_calib_file = config_data["use calibration file"]

        # Determine whether to read information from the calibration data file.
        if is_use_calib_file:
            calib_fp = str(repo_path) + config_data["calibration file path"]

            calib_data = parser_ibm_calib_data(calib_fp)
            config_data["use calibration file"] = False

            for k, v in calib_data.items():
                config_data.update({k: v})
        else:
            pass

        chip_obj = self._create_sc_chip(config_data)

        return chip_obj

    def _parse_chips_info(self, config_data: Dict[str, Any]) -> ChipsNet:
        """
        Read the superconducting quantum multi-chips information from the config file.
        """
        chips_info = config_data["chips"]
        remote_couplings_info = config_data["remote connection"]

        for chip_name, chip_info_detail in chips_info.items():
            # Determine whether to read information from the calibration data file.
            if chip_info_detail["use calibration file"]:
                config_data["chips"][chip_name]["use calibration file"] = False

                # Get the calibration data file path.
                calib_fp = str(repo_path) + chip_info_detail["calibration file path"]

                if self._chip_type == "ibm":
                    calib_data = parser_ibm_calib_data(Path(calib_fp))
                elif self._chip_type == "zcz":
                    calib_data = parser_zcz_calib_data(Path(calib_fp))

                for k, v in calib_data.items():
                    config_data["chips"][chip_name][k] = v
            else:
                pass

        each_chip_commu_qubits_name = self._create_commu_qubit_name_list(
            remote_couplings_info
        )

        # Create the chip(ScChip) list.
        if not self._intra_chip_all2all:
            sc_chip_list, qubits_n_idx = self._create_ScChip_list(
                config_data["chips"], each_chip_commu_qubits_name
            )
        else:
            sc_chip_list, qubits_n_idx = self._create_all2all_ScChip_list(
                config_data["chips"], each_chip_commu_qubits_name
            )

        # Create the remote coupling(RmCoupling) list
        RmCoupling_list = self._create_remote_couplings(
            remote_couplings_info, sc_chip_list
        )

        # Create the chips network
        quantum_chips_net = ChipsNet(sc_chip_list, RmCoupling_list)
        quantum_chips_net.update_qubits_index(qubits_n_idx)

        return quantum_chips_net

    def _create_all2all_ScChip_list(
        self, 
        chips_info: Dict,
        each_chip_commu_qubits_name: Dict[str, List[str]],
    ) -> Tuple[Set[ScChip], bidict]:
        """Create the list of all-to-all quantum chips(ScChip)."""
        sc_chip_list = set()

        chip_idx = 0
        qubit_global_idx = 0
        qubits_n_idx = bidict()

        for chip_name, chip_info in chips_info.items():
            # Create a quantum chip(ScChip).
            chip_obj = ScChip(chip_name, chip_idx)

            physical_qubits_name = chip_info["qubits"]
            single_qubit_gates_fid = chip_info["fidelity"]

            couplings_origin_info = []
            p_qubit_name_pairs = itertools.combinations(physical_qubits_name, 2)
            for qubit_name_pair in p_qubit_name_pairs:
                coupling_info = {
                    "qubit pair": list(qubit_name_pair),
                    "fidelity": 1.0,  # Assuming all-to-all coupling has perfect fidelity
                }
                couplings_origin_info.append(coupling_info)

            # Add the qubits(ScQubit) to the quantum chip(ScChip).
            for physical_qubit_name in physical_qubits_name:
                is_comm_qubit = self._is_commu_qubit(
                    chip_name, physical_qubit_name, each_chip_commu_qubits_name
                )
                qubits_n_idx[physical_qubit_name] = qubit_global_idx

                sc_qubit_obj = ScQubit(
                    qubit_global_idx,
                    chip_idx,
                    is_comm_qubit,
                    single_qubit_gates_fid[physical_qubit_name],
                )
                sc_qubit_obj.set_qubit_name(physical_qubit_name)

                chip_obj.add_qubit(sc_qubit_obj)
                qubit_global_idx += 1
            qubit_list = chip_obj.qubits

            # Add the couplings(ScCoupling) to the quantum chip(ScChip).
            couplings_info = list()
            for coupling in couplings_origin_info:
                coupling_info = coupling["qubit pair"]
                coupling_info.append(coupling["fidelity"])
                couplings_info.append(coupling_info)

            connection_tmp_data = list()
            for ele_info in couplings_info:
                qubit_1_name = ele_info[0]
                qubit_2_name = ele_info[1]
                for qubit in qubit_list:
                    if qubit_1_name == qubit.name:
                        connection_tmp_data.append(qubit)

                    if qubit_2_name == qubit.name:
                        connection_tmp_data.append(qubit)

                connection_tmp_data.append(ele_info[2])
                chip_obj.add_connection(
                    Connection(
                        connection_tmp_data[0],
                        connection_tmp_data[1],
                        connection_tmp_data[2],
                    )
                )
                connection_tmp_data.clear()

            sc_chip_list.add(chip_obj)

            chip_idx += 1

        return sc_chip_list, qubits_n_idx

    def _create_ScChip_list(
        self,
        chips_info: Dict,
        each_chip_commu_qubits_name: Dict[str, List[str]],
    ) -> Tuple[Set[ScChip], bidict]:
        """Create the list of quantum chips(ScChip)."""
        sc_chip_list = set()

        chip_idx = 0
        qubit_global_idx = 0
        qubits_n_idx = bidict()

        for chip_name, chip_info in chips_info.items():
            # Create a quantum chip(ScChip).
            chip_obj = ScChip(chip_name, chip_idx)

            physical_qubits_name = chip_info["qubits"]
            single_qubit_gates_fid = chip_info["fidelity"]
            couplings_origin_info = chip_info["couplings"]

            # Add the qubits(ScQubit) to the quantum chip(ScChip).
            for physical_qubit_name in physical_qubits_name:
                is_comm_qubit = self._is_commu_qubit(
                    chip_name, physical_qubit_name, each_chip_commu_qubits_name
                )
                qubits_n_idx[physical_qubit_name] = qubit_global_idx

                sc_qubit_obj = ScQubit(
                    qubit_global_idx,
                    chip_idx,
                    is_comm_qubit,
                    single_qubit_gates_fid[physical_qubit_name],
                )
                sc_qubit_obj.set_qubit_name(physical_qubit_name)

                chip_obj.add_qubit(sc_qubit_obj)
                qubit_global_idx += 1
            qubit_list = chip_obj.qubits

            # Add the couplings(ScCoupling) to the quantum chip(ScChip).
            couplings_info = list()
            for coupling in couplings_origin_info:
                coupling_info = coupling["qubit pair"]
                coupling_info.append(coupling["fidelity"])
                couplings_info.append(coupling_info)

            connection_tmp_data = list()
            for ele_info in couplings_info:
                qubit_1_name = ele_info[0]
                qubit_2_name = ele_info[1]
                for qubit in qubit_list:
                    if qubit_1_name == qubit.name:
                        connection_tmp_data.append(qubit)

                    if qubit_2_name == qubit.name:
                        connection_tmp_data.append(qubit)

                connection_tmp_data.append(ele_info[2])
                chip_obj.add_connection(
                    Connection(
                        connection_tmp_data[0],
                        connection_tmp_data[1],
                        connection_tmp_data[2],
                    )
                )
                connection_tmp_data.clear()

            sc_chip_list.add(chip_obj)

            chip_idx += 1

        return sc_chip_list, qubits_n_idx

    def _create_sc_chip(self, config_data: Dict[str, Any]) -> ScChip:
        """Create an object of quantum chip(ScChip)."""
        chip_idx = 0
        chip_name = "Chip 0"
        qubit_global_idx = 0
        chip_obj = ScChip(chip_name, chip_idx)

        physical_qubits_name = config_data["qubits"]
        single_qubit_gates_fid = config_data["fidelity"]
        couplings_origin_info = config_data["couplings"]

        # Add the qubits(ScQubit) to the quantum chip(ScChip).
        for physical_qubit_name in physical_qubits_name:
            is_comm_qubit = False
            sc_qubit_obj = ScQubit(
                qubit_global_idx,
                chip_idx,
                is_comm_qubit,
                single_qubit_gates_fid[physical_qubit_name],
            )
            sc_qubit_obj.set_qubit_name(physical_qubit_name)
            chip_obj.add_qubit(sc_qubit_obj)
            qubit_global_idx += 1
        qubit_list = chip_obj.qubits

        # Add the couplings(ScCoupling) to the quantum chip(ScChip).
        couplings_info = list()
        for coupling in couplings_origin_info:
            coupling_info = coupling["qubit pair"]
            coupling_info.append(coupling["fidelity"])
            couplings_info.append(coupling_info)

        connection_tmp_data = list()
        for ele_info in couplings_info:
            qubit_1_name = ele_info[0]
            qubit_2_name = ele_info[1]

            for qubit in qubit_list:
                if qubit_1_name == qubit.name:
                    connection_tmp_data.append(qubit)
                else:
                    pass

                if qubit_2_name == qubit.name:
                    connection_tmp_data.append(qubit)
                else:
                    pass

            connection_tmp_data.append(ele_info[2])
            chip_obj.add_connection(
                Connection(
                    connection_tmp_data[0],
                    connection_tmp_data[1],
                    connection_tmp_data[2],
                )
            )
            connection_tmp_data.clear()

        return chip_obj

    def _create_remote_couplings(
        self, remote_couplings_info: List[Dict[str, str]], chips: List[ScChip]
    ) -> List[ChipsConnection]:
        """Create the list of remote couplings(RmCoupling)."""
        Remote_Couplings = list()

        commu_qubits = list()
        for remote_coupling_info in remote_couplings_info:
            for key, value in remote_coupling_info.items():
                if key != "fidelity":
                    for chip in chips:
                        if key == chip.name:
                            for qubit in chip.qubits:
                                if value == qubit.name:
                                    commu_qubits.append(qubit)

            remote_coupling_obj = ChipsConnection(
                commu_qubit0=commu_qubits[0],
                commu_qubit1=commu_qubits[1],
                fidelity=remote_coupling_info["fidelity"],
            )
            Remote_Couplings.append(remote_coupling_obj)

            commu_qubits.clear()

        return Remote_Couplings

    def _create_commu_qubit_name_list(
        self, remote_couplings_info: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Create remote coupling list using the name of communication qubit"""
        commu_qubits_info = {}

        for remote_coupling in remote_couplings_info:
            chip_name_list = list(remote_coupling.keys())[:2]
            for chip_name in chip_name_list:
                commu_qubit_name = remote_coupling[chip_name]

                if chip_name in commu_qubits_info:
                    if commu_qubit_name not in commu_qubits_info[chip_name]:
                        commu_qubits_info[chip_name].append(commu_qubit_name)
                else:
                    commu_qubits_info.update({chip_name: [commu_qubit_name]})

        return commu_qubits_info

    def _is_commu_qubit(
        self, chip_name: str, qubit_name: str, commu_qubits_info: Dict[str, List[str]]
    ) -> bool:
        """Determine whether a physical qubit is a communication qubit."""
        is_commu_qubit = False

        if (chip_name in commu_qubits_info) and (
            qubit_name in commu_qubits_info[chip_name]
        ):
            is_commu_qubit = True
        else:
            pass

        return is_commu_qubit
