from ansatzes.ansatzes_qiskit import angles_ansatz01_qiskit, ansatz_qiskit_01, ansatz_qiskit_02
import pandas as pd
import numpy as  np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.result import Result

circ = ansatz_qiskit_02(nqubits=3, depth=2)
print(circ)
qc, _ = angles_ansatz01_qiskit(circuit=circ)
print(qc)


