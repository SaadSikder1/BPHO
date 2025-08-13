import numpy as np
import matplotlib.pyplot as plt

# Wavelength range in µm
wavelength_um = np.linspace(0.38, 0.75, 400)  # 380nm to 750nm
wavelength_m = wavelength_um * 1e-6
c = 3e8
frequency_Hz = c / wavelength_m  # Convert wavelength to frequency
frequency_THz = frequency_Hz * 1e-12  # For plotting

# Empirical n(λ) model for water
def n_water(lambda_um):
    return 1.33 + (0.05792105 / (lambda_um**2 - 0.00167917))

n = n_water(wavelength_um)

# Primary rainbow formulas
def theta_primary(n):
    return np.arcsin(np.sqrt((4 - n**2) / 3))

def epsilon_primary(n):
    theta = theta_primary(n)
    return 4 * np.arcsin(np.sin(theta) / n) - 2 * theta

# Secondary rainbow formulas
def theta_secondary(n):
    return np.arcsin(np.sqrt((9 - n**2) / 8))

def epsilon_secondary(n):
    theta = theta_secondary(n)
    return np.pi - 6 * np.arcsin(np.sin(theta) / n) + 2 * theta

# Calculate ε (in degrees)
epsilon1_deg = np.degrees(epsilon_primary(n))
epsilon2_deg = np.degrees(epsilon_secondary(n))

# Color map for rainbow
colors = plt.cm.rainbow(np.linspace(0, 1, len(wavelength_um)))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(frequency_THz, epsilon1_deg, c=colors, s=15, label='Primary Rainbow')
plt.scatter(frequency_THz, epsilon2_deg, c=colors, s=15, label='Secondary Rainbow')

plt.xlabel("Frequency (THz)")
plt.ylabel("ε / deg")
plt.title("Rainbow Elevation ε vs Frequency using Descartes Model (Water)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('11bimage')