import sys
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as  np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SciPyOptimizer
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.result import Result
from qiskit_aer import AerSimulator
#import qat.lang.AQASM as qlm
#from qat.qlmaas import QLMaaSConnection
#from qat.core import Result
#from qat.fermion.circuits import make_ldca_circ, make_general_hwe_circ
sys.path.append("../../")
from utils.utils_ph import create_folder
logger = logging.getLogger('__name__')


def angles_ansatz01_qiskit(circuit, pdf_parameters=None):
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


def submit_circuit_qiskit(qiskit_circuit):#, qiskit_qpu, qiskit_token):
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
    #service = QiskitRuntimeService(channel="ibm_quantum", token=qiskit_token)
    #backend = service.backend(qiskit_qpu)
    backend = AerSimulator()
    nqubits = qiskit_circuit.num_qubits
    #qiskit_circuit.measure_all()

    # Transpile circuit to ISA
    isa_circuit = transpile(qiskit_circuit, backend=backend)
    observable = SparsePauliOp('Z'*nqubits)
    isa_observable = observable.apply_layout(isa_circuit.layout)

    # Submit job using Estimator
    sampler = Sampler(backend)
    vqe = SamplingVQE(sampler=sampler, ansatz=isa_circuit, optimizer=SciPyOptimizer(method='COBYLA'))
    result = vqe.compute_minimum_eigenvalue(operator=isa_observable)
    #job = estimator.run([(isa_circuit, isa_observable)])
    #print(f"Job ID: {job.job_id()}")
    result = result

    return result


def solving_circuit_qiskit(qiskit_state, nqubit, reverse=True):
    """
    Solving a complete Qiskit circuit

    Parameters
    ----------

    qiskit_circuit : Qiskit QuantumCircuit
        Qiskit circuit to solve
    nqubit : int
        number of qubits of the input circuit
    reverse : True
        This is for ordering the state from left to right
        If False the order will be form right to left

    Returns
    _______

    state : pandas DataFrame
        DataFrame with the complete simulation of the circuit
    """
    # if not isinstance(qiskit_state, Result):
        # qiskit_state = qiskit_state.join()
        # time_q_run = float(result.meta_data["simulation_time"])

    pdf_state = proccess_qresults(qiskit_state, nqubit, True)
    # For keep the correct qubit order convention for following
    # computations
    if reverse:
        pdf_state.sort_values('Int', inplace=True)
    # A n-qubit-tensor is prefered for returning
    # state = np.array(pdf_state['Amplitude'])
    # mps_state = state.reshape(tuple(2 for i in range(nqubit)))
    return pdf_state


def ansatz_selector_qiskit(ansatz, **kwargs):
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


class SolveCircuitQiskit:

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
        state = submit_circuit_qiskit(self.circuit, self.qpu)
        self.state = solving_circuit_qiskit(state, self.nqubits)
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
        self.state = submit_circuit_qiskit(self.circuit, self.qpu)
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
            result = job_info.result()
            nqubits = result[0].meas.num_bits
            print(nqubits)
            self.solve_ansatz_time = job_info.usage()
            #state = connection.get_result(jobid)
            state = result.get_statevector()
            self.state = state
            print(self.state)
            if self._save:
                self.save_state()
                self.save_time()
        elif status == "QUEUED":
            print("JobId: {} is pending".format(jobid))
        elif status == "CANCELLED":
            print("JobId: {} was cancelled".format(jobid))
        elif status == "RUNNING":
            print("JobId: {} is running".format(jobid))

    def save_parameters(self):
        """
        Saving Parameters
        """
        self.parameters.to_csv(
            self.filename+"_parameters.csv", sep=";")
    def save_state(self):
        """
        Saving State
        """
        state_for_saving = self.state
        state_for_saving.to_csv(self.filename+"_state.csv", sep=";")
    def save_time(self):
        pdf = pd.DataFrame(
            [self.solve_ansatz_time], index=["solve_ansatz_time"]).T
        pdf.to_csv(self.filename+"_solve_ansatz_time.csv", sep=";")


def run_ansatz_qiskit(**configuration):
    """
    For creating an ansatz and solving it
    """

    nqubits = configuration.get("nqubits", None)
    depth = configuration.get("depth", None)
    ansatz = configuration.get("ansatz", None)
    #qpu_ansatz_name = configuration.get("qpu_ansatz", None)
    save = configuration.get("save", False)
    folder = configuration.get("folder", None)

    # Create Ansatz Circuit
    logger.info("Creating ansatz circuit")
    ansatz_conf = {
        "nqubits" :nqubits,
        "depth" : depth,
    }
    tick = time.time()
    circuit = ansatz_selector_qiskit(ansatz, **ansatz_conf)
    tack = time.time()
    create_ansatz_time = tack - tick
    logger.info("Created ansatz circuit in: %s", create_ansatz_time)
    #from qat.core.console import display
    #display(circuit)

    # Fixing Parameters of the Circuit
    if ansatz == "simple01":
        #If ansatz is simple we use fixed angles
        circuit, pdf_parameters = angles_ansatz01_qiskit(circuit)
    else:
        # For other ansatzes we use random parameters
        parameters = {v_ : 2 * np.pi * np.random.rand() for i_, v_ in enumerate(
            circuit.parameters)}
        # Create the DataFrame with the info
        angles = [k for k, v in parameters.items()]
        values = [v for k, v in parameters.items()]
        # create pdf
        pdf_parameters = pd.DataFrame(
            [angles, values],
            index=['key', 'value']).T
        circuit, _ = angles_ansatz01_qiskit(circuit, pdf_parameters)
    #display(circuit)

    # For creating the folder for saving
    if save:
        folder = create_folder(folder)
        filename = "ansatz_{}_nqubits_{}_depth_{}_qpu_ansatz_{}".format(
            ansatz, nqubits, depth, configuration.get("qpu_ansatz", None))
        filename = folder + filename
    else:
        filename = ""

    # Solving Ansatz
    solve_conf = {
        "qpu" : configuration.get("qpu", None),
        "nqubits" :nqubits,
        "parameters" : pdf_parameters,
        "filename": filename,
        "save": save
    }
    solv_ansatz = SolveCircuitQiskit(circuit, **solve_conf)
    solve = configuration.get("solve", True)
    submit = configuration.get("submit", False)
    if solve:
        logger.info("Solving ansatz circuit")
        solv_ansatz.run()
        solve_ansatz_time = solv_ansatz.solve_ansatz_time
        logger.info("Solved ansatz circuit in: %s", solve_ansatz_time)
        output_dict = {
            "state" : solv_ansatz.state,
            "parameters": pdf_parameters,
            "solve_ansatz_time": solve_ansatz_time,
            "filename" : filename,
            "circuit": circuit
        }
        #print(output_dict["state"])
        return output_dict
    if submit:
        print(solve_conf["filename"])
        logger.info("Ansatz will be submited to QLM")
        solv_ansatz.submit()
        solve_ansatz_time = solv_ansatz.solve_ansatz_time
        return None
    

def getting_job_qiskit(**configuration):
    """
    For getting a job from Qiskit. Configuration need to have following
    keys: nqubits, job_id, save, filename
    """
    #nqubits = configuration.get("nqubits", None)
    job_id = configuration["job_id"]
    save = configuration.get("save", False)
    filename = configuration["filename"]
    logger.info("Job id: %s will be obtained from QLM", job_id)
    solve_conf = {
        "qpu" : None,
        "nqubits" :None,
        "parameters" : None,
        "filename": filename,
        "save": save
    }
    solv_ansatz = SolveCircuitQiskit(None, **solve_conf)
    solv_ansatz.get_job_results(job_id)
    return solv_ansatz.state


if __name__ == "__main__":
    # For sending ansatzes to QLM
    import argparse
    sys.path.append("../../../")
    from qpu.select_qpu import select_qpu
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
        #level=logging.DEBUG
    )
    logger = logging.getLogger('__name__')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nqubits",
        dest="nqubits",
        type=int,
        help="Number of qbits for the ansatz.",
        default=None,
    )
    parser.add_argument(
        "-depth",
        dest="depth",
        type=int,
        help="Depth for ansatz.",
        default=None,
    )
    parser.add_argument(
        "-ansatz",
        dest="ansatz",
        type=str,
        help="Ansatz type: simple01, simple02, lda or hwe.",
        default=None,
    )
    #QPU argument
    parser.add_argument(
        "-qpu_ansatz",
        dest="qpu_ansatz",
        type=str,
        default=None,
        help="QPU for ansatz simulation: " +
            "c, python, linalg, mps, qlmass_linalg, qlmass_mps",
    )
    parser.add_argument(
        "-folder",
        dest="folder",
        type=str,
        default="./",
        help="Path for storing results",
    )
    parser.add_argument(
        "-filename",
        dest="filename",
        type=str,
        default="",
        help="Base Filename for saving. Only Valid with get_job",
    )
    parser.add_argument(
        "--save",
        dest="save",
        default=False,
        action="store_true",
        help="For storing results",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--solve",
        dest="solve",
        default=False,
        action="store_true",
        help="For solving complete ansatz",
    )
    group.add_argument(
        "--submit",
        dest="submit",
        default=False,
        action="store_true",
        help="For submiting ansatz to QLM",
    )
    group.add_argument(
        "--get_job",
        dest="get_job",
        default=False,
        action="store_true",
        help="For getting a job from QLM",
    )
    parser.add_argument(
        "-jobid",
        dest="job_id",
        type=str,
        default=None,
        help="jobid of the QLM job",
    )
    args = parser.parse_args()
    configuration = vars(args)
    qpu_config = {"qpu_type": args.qpu_ansatz}
    configuration.update({"qpu": select_qpu(qpu_config)})
    configuration.update({"qpu_ansatz": args.qpu_ansatz})
    if args.get_job:
        state = getting_job_qiskit(**configuration)
    else:
        output = run_ansatz_qiskit(**configuration)
        if output is not None:
            print(output["state"])
