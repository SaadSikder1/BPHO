import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 299792458  # Speed of light (m/s)
frequencies = np.linspace(400e12, 800e12, 1000)  # Frequency range (Hz)
freq_THz = frequencies / 1e12  # Convert to THz for plotting
wavelengths = c / frequencies  # Wavelengths in meters
wavelengths_um = wavelengths * 1e6  # Convert to microns for Sellmeier

# Sellmeier equation for refractive index of water
def n_water(lambda_um):
    return np.sqrt(
        1 + 0.75831 * lambda_um**2 / (lambda_um**2 - 0.130**2) +
            0.08436 * lambda_um**2 / (lambda_um**2 - 9.896**2)
    )

n = n_water(wavelengths_um)

# Geometry of internal ray paths
theta_p = np.arccos(np.sqrt((n**2 - 1)/3))   # For primary
theta_s = np.arccos(np.sqrt((n**2 - 1)/8))   # For secondary

# Deviation angles
delta_p = np.degrees(4 * theta_p - 2 * np.arcsin(np.sin(theta_p)/n))
delta_s = np.degrees(3 * np.pi - 4 * theta_s + 2 * np.arcsin(np.sin(theta_s)/n))

# Fresnel reflection losses (water to air, partial reflection)
def fresnel_loss(n1, n2, theta_i):
    theta_t = np.arcsin(n1/n2 * np.sin(theta_i))
    rs = ((n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) /
          (n1 * np.cos(theta_i) + n2 * np.cos(theta_t)))**2
    rp = ((n1 * np.cos(theta_t) - n2 * np.cos(theta_i)) /
          (n1 * np.cos(theta_t) + n2 * np.cos(theta_i)))**2
    return 0.5 * (rs + rp)

theta_internal = np.arcsin(np.sin(theta_p) / n)
R = fresnel_loss(n, 1.0, theta_internal)

# Total intensity factors
I_p = (1 - R)**2 * R           # Transmission → reflection → transmission
I_s = (1 - R)**2 * R**2        # 2 reflections

# Normalize intensities for shading
I_p /= np.max(I_p)
I_s /= np.max(I_s)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(freq_THz, delta_p, 'r-', linewidth=2, label='Primary Rainbow Deviation')
plt.plot(freq_THz, delta_s, 'b-', linewidth=2, label='Secondary Rainbow Deviation')
plt.fill_between(freq_THz, delta_p, delta_p + 0.2*I_p, color='red', alpha=0.3, label='Primary Intensity')
plt.fill_between(freq_THz, delta_s, delta_s + 0.2*I_s, color='blue', alpha=0.3, label='Secondary Intensity')

plt.xlabel("Frequency (THz)", fontsize=12)
plt.ylabel("Deviation Angle (degrees)", fontsize=12)
plt.title("Rainbow Deviation Angles vs Frequency\nwith Internal Reflection Losses", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(400, 800)
plt.ylim(130, 180)
plt.tight_layout()
plt.show()
