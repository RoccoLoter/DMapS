import json
import pandas as pd
from typing import Dict
from pathlib import Path
from bidict import bidict
from copy import deepcopy

from global_config import repo_path


def parser_ibm_calib_data(data_fn: Path) -> Dict:
    """Obtain the hardware information from the IBM quantum chip calibration data file.
    Args:
        data_fn  : The path of calibration data file.

    Return:
        json_data: The data that will be written to the json file.
    """
    json_data = dict()

    qubit_list = list()  # the list of qubits
    single_gates_fidelity = dict()  # the single gate fidelity information
    couplings_info = list()  # the couplings information

    qubit_pairs = list()
    dict_qubit_name_label = bidict()
    calibration_data = pd.read_csv(data_fn)
    num_phy_qubits = len(calibration_data)

    # Browse through each line of information in the calibration data file,
    tmp_dict_info = dict()
    tmp_coupling_info = dict()
    qubit_pair = list()
    for i in range(num_phy_qubits):
        qubit_name = calibration_data.loc[i][0]
        dict_qubit_name_label.put(qubit_name, i)

        # 1. Update the list of qubits.
        qubit_list.append(qubit_name)

        # 2. Update the single gate fidelity information.
        tmp_dict_info.update({"sx": (1 - calibration_data.loc[i][10])})
        tmp_dict_info.update({"pauli-x": (1 - calibration_data.loc[i][11])})
        single_gates_fidelity.update({qubit_name: deepcopy(tmp_dict_info)})
        tmp_dict_info.clear()

        # 3. Update the couplings information.
        #    We need to get the coupling and fidelity information from the following string:
        #    "12_15:3.477e-2; 12_13:2.101e-2; 12_10:9.180e-3".
        qubit_rela_coupling_fid = calibration_data.loc[i][12].split("; ")

        for str_coupling_colon_errorRate in qubit_rela_coupling_fid:
            coupling_n_errorRate = str_coupling_colon_errorRate.split(":")
            coupling_related_qubits = coupling_n_errorRate[0].split("_")
            qubit_pair.append("Q" + coupling_related_qubits[0])
            qubit_pair.append("Q" + coupling_related_qubits[1])

            # Determine if the physical connection already exists in "coupling_info".
            if qubit_pair not in qubit_pairs:
                tmp_coupling_info.update(
                    {"fidelity": (1 - eval(coupling_n_errorRate[1]))}
                )
                tmp_coupling_info.update({"qubit pair": deepcopy(qubit_pair)})
                couplings_info.append(deepcopy(tmp_coupling_info))
                qubit_pair.clear()
                tmp_coupling_info.clear()

    json_data.update({"qubits": qubit_list})
    json_data.update({"fidelity": single_gates_fidelity})
    json_data.update({"couplings": couplings_info})

    return json_data


def parser_zcz_calib_data(data_fn: Path) -> Dict:
    """
    Obtain the quantum hardware information from the Zuchongzhi quantum chip calibration data file.
    Args:
        data_fn  : The path of calibration data file.
    """
    json_data = {}

    qubit_list = []  # The physical qubit list.
    single_gates_fid = {}  # The single gates fidelity information.
    coupling_list = []  # The couplings information

    with data_fn.open("r") as df:
        cali_data = json.load(df)
    couplers_map = cali_data["overview"]["coupler_map"]
    used_couplings = cali_data["twoQubitGate"]["czGate"]["gate error"]["qubit_used"]

    # 1. Get the used physical qubits information.
    for coupling in used_couplings:
        for qubit in couplers_map[coupling]:
            if qubit not in qubit_list:
                qubit_list.append(qubit)

    # 2. Get the single gates fidelity information.
    for qubit in qubit_list:
        single_gates_fid[qubit] = 0
    used_qubits = cali_data["qubit"]["singleQubit"]["gate error"]["qubit_used"]
    param_list = cali_data["qubit"]["singleQubit"]["gate error"]["param_list"]
    for i in range(len(used_qubits)):
        single_gates_fid[used_qubits[i]] = {"x/2": 1 - 0.01 * param_list[i]}

    # 3. Get the physical couplings information.
    couplings_param_list = cali_data["twoQubitGate"]["czGate"]["gate error"][
        "param_list"
    ]
    couplings_param_dict = {
        used_couplings[i]: couplings_param_list[i] for i in range(len(used_couplings))
    }

    for coupler_name, qubit_pair in couplers_map.items():
        if coupler_name in used_couplings:
            tmp_dict = {}
            tmp_dict["qubit pair"] = qubit_pair
            tmp_dict["fidelity"] = 1 - 0.01 * couplings_param_dict[coupler_name]
            coupling_list.append(tmp_dict)

    json_data["qubits"] = qubit_list
    json_data["fidelity"] = single_gates_fid
    json_data["couplings"] = coupling_list

    return json_data


def parser_zcz_total_calib_data(data_fn: Path) -> Dict:
    """
    Obtain the quantum hardware information from the Zuchongzhi quantum chip calibration data file.
    Args:
        data_fn  : The path of calibration data file.
    """
    json_data = {}

    qubit_list = []  # The physical qubit list.
    single_gates_fid = {}  # The single gates fidelity information.
    coupling_list = []  # The couplings information

    with data_fn.open("r") as df:
        cali_data = json.load(df)

    # 1. Get the physical qubits information.
    qubit_list = cali_data["overview"]["qubits"]

    # 2. Get the single gates fidelity information.
    for qubit in qubit_list:
        single_gates_fid[qubit] = 0
    used_qubits = cali_data["qubit"]["singleQubit"]["gate error"]["qubit_used"]
    param_list = cali_data["qubit"]["singleQubit"]["gate error"]["param_list"]
    for i in range(len(used_qubits)):
        single_gates_fid[used_qubits[i]] = {"x/2": 1 - 0.01 * param_list[i]}

    # 3. Get the physical couplings information.
    used_couplings = cali_data["twoQubitGate"]["czGate"]["gate error"]["qubit_used"]
    couplings_param_list = cali_data["twoQubitGate"]["czGate"]["gate error"][
        "param_list"
    ]
    couplings_param_dict = {
        used_couplings[i]: couplings_param_list[i] for i in range(len(used_couplings))
    }

    couplers_map = cali_data["overview"]["coupler_map"]
    for coupler_name, qubit_pair in couplers_map.items():
        if coupler_name in used_couplings:
            tmp_dict = {}
            tmp_dict["qubit pair"] = qubit_pair
            tmp_dict["fidelity"] = 1 - 0.01 * couplings_param_dict[coupler_name]
            coupling_list.append(tmp_dict)

    json_data["qubits"] = qubit_list
    json_data["fidelity"] = single_gates_fid
    json_data["couplings"] = coupling_list

    return json_data


def parser_zcz_build_net_info(data_fn: Path) -> Dict:
    """
    Parser the single ZuChongzhi quantum chip calibration data and build the quantum chip network.
    """
    json_data = {}

    with data_fn.open("r") as df:
        config_data = json.load(df)

    json_data["has multiple chips"] = True
    json_data["chips"] = {}
    json_data["remote connection"] = config_data["remote connection"]

    calib_fp = str(repo_path) + config_data["calibration file path"]
    calib_data = parser_zcz_calib_data(Path(calib_fp))

    chip_fid_info = calib_data["fidelity"]
    chip_coupling_info = calib_data["couplings"]
    chips_info = config_data["chips"]
    for chip_name, chip_info in chips_info.items():
        tmp_chip_info = {}
        tmp_chip_info["use calibration file"] = False

        fid_info = {}
        couplings_info = []

        # 1. Get the qubits of virtual quantum chip.
        qubits = chip_info["qubits"]
        tmp_chip_info["qubits"] = qubits

        # 2. Get the single gates fidelity information of virtual quantum chip.
        for qubit, fid in chip_fid_info.items():
            if qubit in qubits:
                fid_info[qubit] = fid
        tmp_chip_info["fidelity"] = fid_info

        # 3. Get the couplings information of virtual quantum chip.
        for coupling in chip_coupling_info:
            qubit_pair = coupling["qubit pair"]
            if qubit_pair[0] in qubits and qubit_pair[1] in qubits:
                couplings_info.append(coupling)
        tmp_chip_info["couplings"] = couplings_info

        json_data["chips"][chip_name] = tmp_chip_info

    return json_data
