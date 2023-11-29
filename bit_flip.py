from qiskit import QuantumRegister
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, IBMQ
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_distribution

if __name__ == '__main__':
    IBMQ.enable_account('')
    provider = IBMQ.get_provider(hub='ibm-q')

    backend = provider.get_backend('ibmq_qasm_simulator')

    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(1, 'c')

    circuit = QuantumCircuit(q, c)

    circuit.cx(q[0], q[1])
    circuit.cx(q[0], q[2])
    # circuit.x(q[0])  # Add this to simulate a bit flip error
    circuit.cx(q[0], q[1])
    circuit.cx(q[0], q[2])
    circuit.ccx(q[2], q[1], q[0])

    circuit.draw(output='mpl',  style="iqp")
    circuit.measure(q[0], c[0])

    sampler = Sampler()
    dist = sampler.run(circuit, shots=1000, seed=42).result().quasi_dists[0]
    plot_distribution(dist.binary_probabilities())
    plt.show()

