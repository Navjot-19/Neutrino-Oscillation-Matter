import numpy as np
import matplotlib.pyplot as plt

# Define the mixing matrix elements and parameters
M11 = M22 = M33 = M44 = M12 = M21 = 0
M13 = M23 = M31 = M32 = 24.4 * 10**-4
M14 = M24 = M41 = M42 = 1
M34 = M43 = 1

M = np.array([
    [M11, M12, M13, M14],
    [M21, M22, M23, M24],
    [M31, M32, M33, M34],
    [M41, M42, M43, M44]
])

theta12 = 0.5 * np.arcsin(np.sqrt(0.846))
theta13 = 0.5 * np.arcsin(np.sqrt(0.093))
theta23 = 0.5 * np.arcsin(np.sqrt(0.92))
theta14 = theta24 = theta34 = 10**-3

phase1 = phase2 = phase3 = 0  # Phases are zero

# Define rotation matrices
W34 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, np.cos(theta34), np.sin(theta34)],
    [0, 0, -np.sin(theta34), np.cos(theta34)]
])

R24 = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta24), 0, np.sin(theta24)],
    [0, 0, 1, 0],
    [0, -np.sin(theta24), 0, np.cos(theta24)]
])

W14 = np.array([
    [np.cos(theta14), 0, 0, np.sin(theta14)],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [-np.sin(theta14), 0, 0, np.cos(theta14)]
])

U23 = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta23), np.sin(theta23), 0],
    [0, -np.sin(theta23), np.cos(theta23), 0],
    [0, 0, 0, 1]
])

U13 = np.array([
    [np.cos(theta13), 0, np.sin(theta13), 0],
    [0, 1, 0, 0],
    [-np.sin(theta13), 0, np.cos(theta13), 0],
    [0, 0, 0, 1]
])

U12 = np.array([
    [np.cos(theta12), np.sin(theta12), 0, 0],
    [-np.sin(theta12), np.cos(theta12), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Construct the mixing matrix U
U = W34 @ R24 @ W14 @ U23 @ U13 @ U12

# Define alpha and beta
alpha = 1
beta = 1
Nmax = 4

# Function to calculate survival probability
def survival_prob(LE):
    sum_real = 0
    for i in range(Nmax - 1):
        for j in range(i + 1, Nmax):
            prod = (U[alpha-1, i] * np.conjugate(U[beta-1, i]) *
                    np.conjugate(U[alpha-1, j]) * U[beta-1, j]) * \
                    np.sin(1.27 * M[i, j] * LE)**2
            sum_real += np.real(prod)
    
    if alpha == beta:
        return 1 - 4 * sum_real
    else:
        return - 4 * sum_real

# Generate data for plotting
LE_values = np.logspace(-1, 1, 500)  # Log scale from 0.1 to 10
P_values = [survival_prob(LE) for LE in LE_values]

# Plot the survival probability
plt.figure(figsize=(10, 6))
plt.plot(LE_values, P_values, label=r'$P(\nu_e \to \nu_e)$', color='blue')
plt.xscale('log')
plt.xlabel('Length/Energy in km/GeV')
plt.ylabel(r'$P(\nu_e \to \nu_e)$')
plt.title(r'Survival Probability of $\nu_e$ in Vacuum')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
