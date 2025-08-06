from comparison_algorithms.pytket_dqc.distributors.partitioning_heterogeneous import (
    PartitioningAnnealing,
    PartitioningHeterogeneous,
    PartitioningHeterogeneousEmbedding
)
from comparison_algorithms.pytket_dqc.networks.nisq_network import NISQNetwork
from comparison_algorithms.pytket_dqc.utils.gateset import DQCPass
from pytket import Circuit, OpType


network = NISQNetwork([[0,1], [0,2]], {0:[0], 1:[1], 2:[2]})
circ = Circuit(2)
circ.add_gate(OpType.CU1, 1.0, [0, 1]).H(0).Rz(0.3,0).H(0).add_gate(OpType.CU1, 1.0, [0, 1])
# circ = Circuit(7).CX(0,1).CX(1,2).CX(2,3).CX(3,4).CX(4,5).CX(5, 6)
# pass_circ = DQCPass().apply(circ)

distribution = PartitioningHeterogeneous().distribute(circ, network)