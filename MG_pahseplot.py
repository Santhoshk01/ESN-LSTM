import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Mackey-Glass system
beta = 0.2
gamma = 0.1
n = 4000

# Define the tau values to plot
tau_values = [3, 5, 7, 11, 13, 17]

# Create a figure with 2x3 subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Initialize the Mackey-Glass system
x = np.zeros((len(tau_values), n))
x[:, 0] = 1.5

# Update the system using the Mackey-Glass equation for each tau value
for i, tau in enumerate(tau_values):
    for j in range(0, n-1):
        x[i, j+1] = x[i, j] + (beta * x[i, j-tau]) / (1 + x[i, j-tau]**10) - gamma * x[i, j]

    # Plot the phase portrait for each tau value, skipping the first 2*tau values
    row = i // 3
    col = i % 3
    axes[row, col].plot(x[i, 2*tau:-tau], x[i, 3*tau:], 'b-', linewidth=0.6)
    axes[row, col].set_title(f"Tau = {tau}")
    axes[row, col].set_xlabel('x(t)')
    axes[row, col].set_ylabel('x(t-τ)')
    axes[row, col].grid(False)

plt.suptitle('Phase Portraits of Mackey-Glass System with Various Tau')
plt.tight_layout()
plt.show()
