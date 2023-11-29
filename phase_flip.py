from matplotlib import pyplot as plt
from qiskit import QuantumRegister, Aer
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit,IBMQ
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_distribution, plot_state_qsphere
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

IBMQ.enable_account('')
provider = IBMQ.get_provider(hub='ibm-q')

backend = Aer.get_backend('qasm_simulator')


qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

#circuit.x(qreg_q[0]) # 1
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.h(qreg_q[2])
circuit.h(qreg_q[1])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.h(qreg_q[2])
circuit.z(qreg_q[0])
circuit.h(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[0])
psi = Statevector(circuit)
circuit.measure(qreg_q[0], creg_c[0])
circuit.barrier(qreg_q)

sampler = Sampler()
dist = sampler.run(circuit, shots=1000, seed=42).result().quasi_dists[0]
plot_distribution(dist.binary_probabilities())
circuit.draw(output='mpl', style="iqp")
plot_state_qsphere(psi, show_state_phases=True)
plt.show()
