from ansatzes.ansatzes_qiskit import angles_ansatz01_qiskit, ansatz_qiskit_01, ansatz_qiskit_02, submit_circuit_qiskit
#from parent_hamiltonian.parent_hamiltonian import PH
import pandas as pd
import numpy as  np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.result import Result
from qiskit_aer import AerSimulator

qc = ansatz_qiskit_02(nqubits=3, depth=2)
#print(circ)
#qc, _ = angles_ansatz01_qiskit(circuit=circ)
print(qc)
result = submit_circuit_qiskit(qc)
print(f"  > Expectation value: {result.eigenvalue}")
print(f"  > Eigenstate: {result.eigenstate}")
#counts = result[0].get_counts()
#print(result[0].data.items())
#print(result[0].data.get_counts())



