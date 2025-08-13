import numpy as np
import matplotlib.pyplot as plt

# Constants
n = 1.33  # Refractive index of water

# Incident angles (degrees)
theta_deg = np.linspace(0.01, 90, 1000)
theta_rad = np.radians(theta_deg)

# Refraction angle (radians)
theta_r = np.arcsin(np.sin(theta_rad) / n)

# Primary rainbow deviation and elevation
D_primary = 2 * theta_deg - 4 * np.degrees(theta_r) + 180
epsilon_primary = 180 - D_primary

# Secondary rainbow deviation and elevation
D_secondary = 6 * np.degrees(theta_r) - 2 * theta_deg
epsilon_secondary = 180 - D_secondary

# Find where secondary elevation is visible
valid_secondary = (epsilon_secondary > 0) & (epsilon_secondary < 180)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(theta_deg, epsilon_primary, label='Primary Rainbow (1 reflection)', color='blue')
plt.plot(theta_deg[valid_secondary], epsilon_secondary[valid_secondary],
         label='Secondary Rainbow (2 reflections)', color='orange')

# Guide lines
plt.axhline(y=42, color='blue', linestyle='--', linewidth=0.8, label='Primary Max ~42°')
plt.axhline(y=51, color='orange', linestyle='--', linewidth=0.8, label='Secondary Max ~51°')

# Labels
plt.xlabel('Angle of Incidence θ (degrees)')
plt.ylabel('Rainbow Elevation ε (degrees)')
plt.title('Rainbow Elevation (ε) vs Angle of Incidence (θ)')
plt.ylim(0, 180)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('11a.png')