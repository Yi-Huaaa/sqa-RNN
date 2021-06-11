"""
* Given the same exp setting in paper: 
    number of spin: 32, 64, 128
    magnetic field: 40 // Bx
* Main purpose:
    Computes the matrix element of the open Ising Hamiltonian for a given state sigmap
    i.e. Computes the ground state energy of the 1D Ising energy model

* input:
    Jz:         np.ndarray of shape (N), respectively, and dtype=float:
                Ising parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
                A sample of spins can be fed here.
    Bx:         Scale of the transvers magnetic field (N)
* return: 2-tuple of type (np.ndarray,np.ndarray)
        sigmas:         np.ndarray of dtype=int and shape (?,N)
                         the states for which there exist non-zero matrix elements for given sigmap
        matrixelements: np.ndarray of dtype=float and shape (?)
                         the non-zero matrix elements
"""
#the diagonal part is simply the sum of all Sz-Sz interactions
import numpy as np
def IsingMatrixElements(Jz, Bx, sigmap):
    diag = 0
    
    sigmas = []
    matrix_elements = []
    N = Jz.shape[0]
    
    for site in range(N-1):
        if sigmap[site] == sigmap[site+1]: #if the two neighouring spins are the same (We use open Boundary Conditions)
            diag-=Jz[site] #add a negative energy contribution (We use ferromagnetic couplings)
        else:
            diag+=Jz[site]
            
    matrix_elements.append(diag)
    sigmas.append(sigmap)
    
    #off-diagonal part (For the transverse Ising Model)
    for site in range(N):
        if Bx[site] != 0:
            sig = np.copy(sigmap)
            sig[site]=np.abs(1-sig[site])
            matrix_elements.append(-Bx[site])
            sigmas.append(sig)    

    return np.array(sigmas),np.array(matrix_elements)

def ED_1DTFIM(N = 10, h = 1):
    """
    Returns a tuple (eta,U)
        eta = a list of energy eigenvalues.
        U = a list of energy eigenvectors
    """
    Jz=+np.ones(N)
    Bx=+h*np.ones(N)

    basis = []
    #Generate a z-basis
    for i in range(2**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in bin(i)[2:]])
        l = len(a)
        basis_temp[N-l:] = a
      
        basis.append(basis_temp)
    basis = np.array(basis)
      
    H = np.zeros((basis.shape[0],basis.shape[0])) #prepare the hamiltonian
    for n in range(basis.shape[0]):
        sigmas,elements = IsingMatrixElements(Jz,Bx,basis[n])
        for m in range(sigmas.shape[0]):
            for b in range(basis.shape[0]):
                if np.all(basis[b,:] == sigmas[m,:]):
                    H[n,b] = elements[m]
                    break
    eta,U = np.linalg.eigh(H) #diagonalize
    return eta,U

# Calculate 1D Ising model, ground state energy
# N: spin energy, h = Bx
h = 40
N = 32
eta, U = ED_1DTFIM(N, h)
print('Spin number =', N ,';The ground state energy is:', min(eta))
N = 64
eta, U = ED_1DTFIM(N, h)
print('Spin number =', N ,';The ground state energy is:', min(eta))
N = 128
eta, U = ED_1DTFIM(N, h)
print('Spin number =', N ,';The ground state energy is:', min(eta))
#E_exact = min(eta)