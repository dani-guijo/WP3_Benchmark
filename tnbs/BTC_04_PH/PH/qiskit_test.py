from ansatzes.ansatzes_qiskit import angles_ansatz01_qiskit, ansatz_qiskit_01, ansatz_qiskit_02, solve_circuit_qiskit, SolveCircuitQiskit
from parent_hamiltonian.parent_hamiltonian import PH
import pandas as pd
import numpy as  np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.result import Result
from qiskit_aer import AerSimulator

qc = ansatz_qiskit_02(nqubits=3, depth=2)

# Use SamplingVQE to solve the circuit
print(qc)
scq = SolveCircuitQiskit(qc)
scq.run()
state = list(scq.state['Amplitude'])

print(f"  > Eigenstate: {state}")

# Compute Naive PH
n_ph = PH(state)
n_ph.naive_ph()
print(f"  > Naive PH shape: {n_ph.rho.shape}") # Must be (2^N,2^N)
n_pdf = n_ph.pauli_pdf
print(n_pdf.head())

# Compute Local PH
l_ph = PH(state)
l_ph.local_ph()
print(f"  > Naive PH length: {len(l_ph.reduced_rho)}") # Must be N
l_pdf = l_ph.pauli_pdf
print(l_pdf.head())
