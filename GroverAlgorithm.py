from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, IBMQ
from qiskit.circuit.library import CXGate, ZGate, MCMT
from qiskit.visualization import plot_distribution, plot_state_qsphere
from matplotlib import pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler
import numpy as np

def Oracle(marked_states): # Oracle function to mark the target bit-strings
    
    if not isinstance(marked_states, list): 
        marked_states = [marked_states]

    for target in marked_states: 
        
        rev_target = target[::-1] 
        zero_inds = [ind for ind in range(len(target)) if rev_target.startswith("0", ind)] # Find the indices of the |0> states in the target bit-string
        
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls) to the circuit if the target qubit is in the |0> state
        
        if zero_inds:
            circuit.x(zero_inds)
        
        circuit.compose(MCMT(ZGate(), len(target) - 1, 1), inplace=True) # Apply the multi-controlled Z-gate with the target qubit as the control and all other qubits as targets 
        
        if zero_inds:
            circuit.x(zero_inds)
    
            
def Grover(marked_states): # Grover operator to apply the oracle function and the diffusion operator
    
    Oracle(marked_states) # Apply the oracle function to the circuit
    circuit.barrier()  # barrier to separate the oracle function from the rest of the circuit

    for i in range(len(qreg_q)): # Apply Hadamard gates and X-gates to all qubits 
        circuit.h(qreg_q[i])
        circuit.x(qreg_q[i])


    circuit.h(len(qreg_q)-1)
    
    circuit.compose(MCMT(CXGate(), len(qreg_q) - 1, 1), inplace=True) # Apply the multi-controlled X-gate with the target qubit as the control and all other qubits as targets 

    circuit.h(len(qreg_q)-1)

    for i in range(len(qreg_q)): # Apply Hadamard gates and X-gates to all qubits and add a barrier to separate the iterations
        circuit.x(qreg_q[i])
        circuit.h(qreg_q[i])
    



if __name__ == "__main__": # Main function to run the Grover algorithm on the simulator and plot the results
    IBMQ.enable_account('')
    provider = IBMQ.get_provider(hub='ibm-q')

    backend = Aer.get_backend('qasm_simulator')
    
    

    bits = 2 # Number of qubits and classical bits in the circuit
    qreg_q = QuantumRegister(bits, 'q')
    creg_c = ClassicalRegister(bits, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    marked_states = ["11"] # List of bit-strings to mark
    
    θ = np.sqrt(len(marked_states) / 2**len(qreg_q), dtype=np.float32) # Calculate the amplitude of the marked states
    R = round((np.arccos(θ)) / (2 * np.arcsin(θ))) # Calculate the number of iterations to apply the Grover operator for optimal performance

    for i in range(len(qreg_q)): # Apply Hadamard gates to all qubits
        circuit.h(qreg_q[i])
    
    for k in range(R): # Apply the Grover iteration R times where R is the number of iterations calculated above
        circuit.barrier() # Barrier to separate the iterations
        Grover(marked_states)
        circuit.barrier() # Barrier to separate the iterations
        
    psi = Statevector(circuit) 
    for i in range(len(qreg_q)): # Measure all qubits
        circuit.measure(qreg_q[i], creg_c[i])


    sampler = Sampler()
    dist = sampler.run(circuit, shots=1000, seed=42).result().quasi_dists[0]
    
    plot_distribution(dist.binary_probabilities(), number_to_keep=len(marked_states)+1)
    plot_state_qsphere(psi, show_state_phases=True)
    
    circuit.draw(output='mpl', style="iqp")
    
    plt.show()
