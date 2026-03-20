import sys
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as  np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
import qat.lang.AQASM as qlm
from qat.qlmaas import QLMaaSConnection
from qat.core import Result
from qat.fermion.circuits import make_ldca_circ, make_general_hwe_circ
sys.path.append("../../")
from PH.utils.utils_ph import create_folder
logger = logging.getLogger('__name__')


def angles_ansatz01(circuit, pdf_parameters=None):
    """
    Create the angles for ansatz01

    Parameters
    ----------

    circuit : Qiskit QuantumCircuit
        Qiskit circuit with the parametrized ansatzes
    parameters : pandas DataFrame
        For providing the parameters to the circuit. If None is provide
        the parameters are set using som formula

    Returns
    _______

    circuit : Qiskit QuantumCircuit
        Qiskit circuit with the parameters fixed
    pdf_parameters : pandas DataFrame
        DataFrame with the values of the parameters
    """
    if pdf_parameters is None:
        parameter_name = circuit.parameters
        # Computing number of layers
        n_layers = len(parameter_name) // 2
        # Setting delta_theta
        theta = np.pi/4.0
        delta_theta = theta / (n_layers + 1)
        parameters = {v_ : (i_+1) * delta_theta \
            for i_, v_ in enumerate(parameter_name)}
        angles = [k for k, v in parameters.items()]
        values = [v for k, v in parameters.items()]
        # create pdf
        pdf_parameters = pd.DataFrame(
            [angles, values],
            index=['key', 'value']).T
    else:
        if isinstance(pdf_parameters, pd.core.frame.DataFrame):
            # Formating Parameters
            parameters = {k:v for k, v in zip(
                pdf_parameters['key'], pdf_parameters['value'])}
        else:
            raise ValueError("pdf_parameters MUST BE a DataFrame")
    circuit = circuit.assign_parameters(parameters)
    return circuit, pdf_parameters


def ansatz_qiskit_01(nqubits=7, depth=3):
    """
    Implements Qiskit version of the Parent Hamiltonian Ansatz using
    parametric Circuit

    Parameters
    ----------

    nqubits: int
        number of qubits for the ansatz
    depth : int
        number of layers for the parametric circuit

    Returns
    _______

    circuit : Qiskit QuantumCircuit
        Qiskit circuit with the ansatz implementation in parametric format
    theta : list
        list with the name of the variables of the circuit
    """

    circuit = QuantumCircuit(nqubits)

    #Parameters for the PQC
    theta = [
        Parameter("theta_{}".format(i)) for i in range(2*depth)
    ]

    for d_ in range(0, 2*depth, 2):
        for i in range(nqubits):
            circuit.rx(theta[d_], i)
        for i in range(nqubits-1):
            circuit.cz(i, i+1)
        circuit.cz(nqubits-1, 0)
        for i in range(nqubits):
            circuit.rz(theta[d_+1], i)
    theta = [th.name for th in theta]
    return circuit


def ansatz_qiskit_02(nqubits, depth=3):
    """
    Implements Qiskit version of the Parent Hamiltonian Ansatz using
    parametric Circuit

    Parameters
    ----------

    nqubits: int
        number of qubits for the ansatz
    depth : int
        number of layers for the parametric circuit

    Returns
    _______

    circuit : Qiskit QuantumCircuit
        Qiskit circuit with the ansatz implementation in parametric format
    theta : list
        list with the name of the variables of the circuit
    """

    circuit = QuantumCircuit(nqubits)

    #Parameters for the PQC
    #theta = [
    #    qprog.new_var(float, "\\theta_{}".format(i)) for i in range(2*depth)
    #]

    theta = []
    indice = 0
    for d_ in range(0, 2*depth, 2):
        for i in range(nqubits):
            step = Parameter("theta_{}".format(indice))
            circuit.rx(step, i)
            theta.append(step)
            indice = indice + 1
        for i in range(nqubits-1):
            circuit.cz(i, i+1)
        circuit.cz(nqubits-1, 0)
        for i in range(nqubits):
            step = Parameter("theta_{}".format(indice))
            circuit.rz(step, i)
            indice = indice + 1
            theta.append(step)
    return circuit


def proccess_qresults(result, qubits, complete=True):
    """
    Post Process a Qiskit results for creating a pandas DataFrame

    Parameters
    ----------

    result : Qiskit results from a Qiskit qpu.
        returned object from a qpu submit
    qubits : int
        number of qubits
    complete : bool
        for return the complete basis state.
    """

    # Process the results
    if complete:
        states = []
        list_int = []
        list_int_lsb = []
        for i in range(2**qubits):
            reversed_i = int("{:0{width}b}".format(i, width=qubits)[::-1], 2)
            list_int.append(reversed_i)
            list_int_lsb.append(i)
            states.append("|" + bin(i)[2:].zfill(qubits) + ">")
        probability = np.zeros(2**qubits)
        amplitude = np.zeros(2**qubits, dtype=np.complex_)
        for samples in result:
            probability[samples.state.lsb_int] = samples.probability
            amplitude[samples.state.lsb_int] = samples.amplitude

        pdf = pd.DataFrame(
            {
                "States": states,
                "Int_lsb": list_int_lsb,
                "Probability": probability,
                "Amplitude": amplitude,
                "Int": list_int,
            }
        )
    else:
        list_for_results = []
        for sample in result:
            list_for_results.append([
                sample.state, sample.state.lsb_int, sample.probability,
                sample.amplitude, sample.state.int,
            ])

        pdf = pd.DataFrame(
            list_for_results,
            columns=['States', "Int_lsb", "Probability", "Amplitude", "Int"]
        )
        pdf.sort_values(["Int_lsb"], inplace=True)
    return pdf


def submit_circuit(qiskit_circuit, qiskit_qpu, qiskit_token):
    """
    Solving a complete Qiskit circuit

    Parameters
    ----------

    qiskit_circuit : Qiskit QuantumCircuit
        Qiskit circuit to solve
    qiskit_qpu : Qiskit QPU
        Qiskit QPU for solving the circuit
    """
    # Creating the qlm_job
    service = QiskitRuntimeService(channel="ibm_quantum", token=qiskit_token)
    backend = service.backend(qiskit_qpu)
    qiskit_circuit.measure_all()

    # Transpile circuit to ISA
    isa_circuit = transpile(qiskit_circuit, backend=backend)

    # Submit job using Sampler
    sampler = Sampler(backend)
    job = sampler.run([(isa_circuit,)])
    print(f"Job ID: {job.job_id()}")
    result = job.result()

    return result


def solving_circuit(qlm_state, nqubit, reverse=True):
    """
    Solving a complete Qiskit circuit

    Parameters
    ----------

    qiskit_circuit : QLM circuit
        qlm circuit to solve
    nqubit : int
        number of qubits of the input circuit
    qlm_qpu : QLM qpu
        QLM qpu for solving the circuit
    reverse : True
        This is for ordering the state from left to right
        If False the order will be form right to left

    Returns
    _______

    state : pandas DataFrame
        DataFrame with the complete simulation of the circuit
    """
    if not isinstance(qlm_state, Result):
        qlm_state = qlm_state.join()
        # time_q_run = float(result.meta_data["simulation_time"])

    pdf_state = proccess_qresults(qlm_state, nqubit, True)
    # For keep the correct qubit order convention for following
    # computations
    if reverse:
        pdf_state.sort_values('Int', inplace=True)
    # A n-qubit-tensor is prefered for returning
    # state = np.array(pdf_state['Amplitude'])
    # mps_state = state.reshape(tuple(2 for i in range(nqubit)))
    return pdf_state


def ansatz_selector(ansatz, **kwargs):
    """
    Function for selecting an ansatz

    Parameters
    ----------

    ansatz : text
        The desired ansatz
    kwargs : keyword arguments
        Different keyword arguments for configuring the ansazt, like
        nqubits or depth

    Returns
    _______

    circuit : Qiskit QuantumCircuit
        The Qiskit circuit circuit implementation of the input ansatz
    """


    nqubits = kwargs.get("nqubits")
    if nqubits is None:
        text = "nqubits can not be none"
        raise ValueError(text)
    depth = kwargs.get("depth")
    if depth is None:
        text = "depth can not be none"
        raise ValueError(text)

    if ansatz == "simple01":
        circuit = ansatz_qiskit_01(nqubits=nqubits, depth=depth)
    if ansatz == "simple02":
        circuit = ansatz_qiskit_02(nqubits=nqubits, depth=depth)
    if ansatz == "lda":
        circuit = make_ldca_circ(nqubits, ncycles=depth)
    if ansatz == "hwe":
        circuit = make_general_hwe_circ(nqubits, n_cycles=depth)
    else:
        text = "ansatz MUST BE simple01, simple02, lda or hwe"
        raise ValueError(text)
    return circuit


class SolveCircuit:

    def __init__(self, circuit, **kwargs):
        """

        Method for initializing the class

        """
        self.circuit = circuit
        self.parameters = kwargs.get("parameters", None)
        self.nqubits = kwargs.get("nqubits", None)

        # For Saving
        self._save = kwargs.get("save", False)
        self.filename = kwargs.get("filename", None)

        # Set the QPU to use
        self.qpu = kwargs.get("qpu", None)

        # For Storing Results
        self.state = None
        self.solve_ansatz_time = None

    def run(self):
        """
        Solve Circuit
        """
        tick = time.time()
        state = submit_circuit(self.circuit, self.qpu)
        self.state = solving_circuit(state, self.nqubits)
        tack = time.time()
        self.solve_ansatz_time = tack - tick
        if self._save:
            self.save_state()
            self.save_parameters()
            self.save_time()

    def submit(self):
        """
        Submit circuit
        """
        #self.circuit = self.circuit(**self.parameters)
        self.state = submit_circuit(self.circuit, self.qpu)
        if self._save:
            self.save_parameters()

    def get_job_results(self, jobid, qiskit_token):
        """
        Given a Jobid retrieve the result and procces output
        """
        # Open QiskitRuntimeService connection
        service = QiskitRuntimeService(channel="ibm_quantum", token=qiskit_token)
        # Get Info of the job
        job_info = service.job(jobid)
        print(job_info)
        status = job_info.status()

        if status == "DONE":
            #Work done
            nqubits = job_info.resources[0].nbqbits
            print(nqubits)
            end = datetime.strptime(
                job_info.ending_date.rsplit(".")[0],
                "%Y-%m-%d %H:%M:%S")
            start = datetime.strptime(
                job_info.starting_date.rsplit(".")[0],
                "%Y-%m-%d %H:%M:%S")
            elapsed = end - start
            elapsed = elapsed.total_seconds()
            self.solve_ansatz_time = elapsed
            #state = connection.get_result(jobid)
            state = service.get_job(jobid)
            self.state = solving_circuit(state, nqubits)
            print(self.state)
            if self._save:
                self.save_state()
                self.save_time()
        elif status == 1:
            print("JobId: {} is pending".format(jobid))
        elif status == 4:
            print("JobId: {} was cancelled".format(jobid))
        elif status == 2:
            print("JobId: {} is running".format(jobid))
