import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha_deg = 45
alpha = np.radians(alpha_deg)
f_THz = 542.5  # THz
c = 3e8  # m/s

# Calculate wavelength
wavelength_nm = (c / (f_THz * 1e12)) * 1e9  # in nm

# Cauchy approximation for glass
def refractive_index_glass(wavelength_nm):
    wl = wavelength_nm / 1000  # microns
    B = 1.5046
    C = 0.00420
    return B + C / (wl**2)

n_glass = refractive_index_glass(wavelength_nm)
n_air = 1.0

# Angles of incidence (degrees)
theta_i_deg = np.linspace(0, 60, 500)
theta_i = np.radians(theta_i_deg)

# Snell's law: air → glass
sin_r1 = n_air / n_glass * np.sin(theta_i)
valid_r1 = np.abs(sin_r1) <= 1
r1 = np.empty_like(theta_i)
r1[:] = np.nan
r1[valid_r1] = np.arcsin(sin_r1[valid_r1])

# Inside prism: apply geometry for deviation
# Second angle of incidence at internal surface
r2 = alpha - r1  # geometry: interior angle of triangle
sin_r2_exit = n_glass / n_air * np.sin(r2)

valid_r2 = (np.abs(sin_r2_exit) <= 1)
theta_t = np.empty_like(theta_i)
theta_t[:] = np.nan
theta_t[valid_r1 & valid_r2] = np.arcsin(sin_r2_exit[valid_r1 & valid_r2])

# Total deviation = theta_i + theta_t - alpha
transmission_angle_deg = np.degrees(theta_t + theta_i - alpha)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(theta_i_deg, transmission_angle_deg, color='blue')
plt.axhline(y=5.787, color='red', linestyle='--', label=r'$\theta_{max}$ = 5.787°')
plt.xlabel("Incident Angle (degrees)")
plt.ylabel("Transmission Angle (degrees)")
plt.title("Transmission Angle vs Incident Angle\n" +
          f"α = {alpha_deg}°, f = {f_THz} THz, λ ≈ {int(wavelength_nm)} nm")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('task12aimage.png')