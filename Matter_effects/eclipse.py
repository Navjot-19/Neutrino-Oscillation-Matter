import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigh

# Define the mixing angles
theta12 = 0.5 * np.arcsin(np.sqrt(0.846))
theta13 = 0.5 * np.arcsin(np.sqrt(0.10))
theta23 = 0.5 * np.arcsin(np.sqrt(0.97))

# Define the phases (set to zero for now)
phase1 = np.exp(1j * 0)
phase2 = np.exp(1j * 0)
phase3 = np.exp(1j * 0)
Phase = np.array([[1, 0, 0], 
                  [0, phase1, 0], 
                  [0, 0, phase2]])

# Define the PMNS matrix
U23 = np.array([[1, 0, 0], 
                [0, np.cos(theta23), np.sin(theta23)], 
                [0, -np.sin(theta23), np.cos(theta23)]])

U13 = np.array([[np.cos(theta13), 0, np.sin(theta13)], 
                [0, 1, 0], 
                [-np.sin(theta13), 0, np.cos(theta13)]])

U12 = np.array([[np.cos(theta12), np.sin(theta12), 0], 
                [-np.sin(theta12), np.cos(theta12), 0], 
                [0, 0, 1]])

U = U23 @ U13 @ U12 @ Phase
print("PMNS Matrix U:\n", U)

# Define the diagonal mass matrices
m11, m22, m33 = 1.0e-5, 1.1e-5, 2.44e-3
SquareMass = np.diag([m11, m22, m33])

# Wolfenstein constant for matter effects
En = 100  # Energy in GeV
Ax = 2 * np.sqrt(2) * 1.166e-5 * 160 * 4.36 * 1 / 0.938 * En
A = np.zeros((3, 3))
A[0, 0] = Ax
print("Matter potential A:\n", A)

# Compute the effective mass matrix in matter
Mm = U @ SquareMass @ U.conj().T + A
eigenvalues, eigenvectors = eigh(Mm)
print("Eigenvalues in matter:\n", eigenvalues)
print("Eigenvectors in matter:\n", eigenvectors)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Define the matter delta mass squared matrix
DelM = np.diag(eigenvalues)
print("Delta Mass Squared in Matter:\n", DelM)

# Survival Probability Calculation
alpha = 1  # for electron neutrino
beta = 1   # for electron neutrino
Nmax = 3

def Peu(LE):
    sum_real = 0
    sum_imag = 0
    for i in range(Nmax-1):
        for j in range(i+1, Nmax):
            prod = (eigenvectors[alpha-1, i] * np.conj(eigenvectors[beta-1, i]) * 
                    np.conj(eigenvectors[alpha-1, j]) * eigenvectors[beta-1, j])
            sum_real += np.real(prod) * (np.sin(1.27 * (eigenvalues[i] - eigenvalues[j]) * LE))**2
            sum_imag += np.imag(prod) * (np.sin(1.27 * (eigenvalues[i] - eigenvalues[j]) * LE))**2
    
    return (1 if alpha == beta else 0) - 4 * sum_real + 2 * sum_imag

# Plot Survival Probability
LE_values = np.logspace(0, 5, 500)
prob_values = [Peu(LE) for LE in LE_values]

plt.figure(figsize=(10, 6))
plt.plot(LE_values, prob_values, label=r'$P(\nu_e \to \nu_e)$', color='blue')
plt.xscale('log')
plt.xlabel('Length/Energy (km/GeV)')
plt.ylabel(r'$P(\nu_e \to \nu_e)$')
plt.title('Survival Probability of $\\nu_e$ in Matter')
plt.grid(True)
plt.legend()
plt.show()
