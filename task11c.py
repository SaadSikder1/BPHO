import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------
def fit_cauchy_three_points(points_nm_n):
    """
    Fit n(λ) = A + B/λ^2 + C/λ^4 (λ in µm) to three (λ_nm, n) points.
    Returns A, B, C.
    """
    lam_nm = np.array([p[0] for p in points_nm_n], dtype=float)
    nvals  = np.array([p[1] for p in points_nm_n], dtype=float)
    lam_um = lam_nm * 1e-3
    X = np.vstack([np.ones_like(lam_um), 1/lam_um**2, 1/lam_um**4]).T
    # Solve exactly (3 equations, 3 unknowns)
    A, B, C = np.linalg.solve(X, nvals)
    return A, B, C

def n_water_cauchy(lam_um, A, B, C):
    """Refractive index via the fitted 3-term Cauchy model."""
    return A + B/lam_um**2 + C/lam_um**4

def r_internal_at_rainbow(n, m):
    """
    Closed-form internal refraction angle r (in radians) for rainbow order m (m=1 primary, m=2 secondary).
    Derived from Snell + stationary-deviation condition.
        cos^2 r = (n^2 - 1) / [ n^2 * (1 - 1/(m+1)^2) ]
    """
    A = 1.0 - 1.0/((m+1)**2)
    cos2r = (n**2 - 1.0)/(n**2 * A)
    # Numerical safety clamp
    cos2r = np.clip(cos2r, 0.0, 1.0)
    r = np.arccos(np.sqrt(cos2r))
    return r

def critical_angle_inside(n):
    """Critical angle (in radians) for rays inside water impinging on the water-air interface."""
    x = 1.0/n
    x = np.clip(x, -1.0, 1.0)
    return np.arcsin(x)

# Simple wavelength (nm) to RGB mapping (approximate human-perceived colors)
# Based on a common analytic approximation adapted from Dan Bruton's algorithm.
def wavelength_to_rgb(wavelength_nm, gamma=0.8):
    w = float(wavelength_nm)
    if w < 380 or w > 780:
        return (0,0,0)
    if w < 440:
        r, g, b = -(w-440)/(440-380), 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w-440)/(490-440), 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w-510)/(510-490)
    elif w < 580:
        r, g, b = (w-510)/(580-510), 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w-645)/(645-580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    # Intensity correction near vision limits
    if   w < 420: factor = 0.3 + 0.7*(w-380)/(420-380)
    elif w < 701: factor = 1.0
    else:         factor = 0.3 + 0.7*(780-w)/(780-700)

    def adj(c):
        if c == 0.0: return 0.0
        return factor * (c ** gamma)

    return (adj(r), adj(g), adj(b))

# -------------------------
# Dispersion fit
# -------------------------
# Anchor points: (Fraunhofer C, d, F lines) typical room-temperature water values
anchor_points = [
    (656.272, 1.3310),  # C (656.3 nm)
    (587.562, 1.3330),  # d (587.6 nm)
    (486.134, 1.3371),  # F (486.1 nm)
]
A, B, C = fit_cauchy_three_points(anchor_points)

# -------------------------
# Frequency grid and conversions
# -------------------------
c = 299_792_458.0  # m/s
f_THz = np.linspace(400.0, 780.0, 241)  # 400–780 THz
f_Hz  = f_THz * 1e12
lam_m = c / f_Hz
lam_nm = lam_m * 1e9
lam_um = lam_nm * 1e-3

# -------------------------
# Compute n(λ), angles r for m=1 (primary) and m=2 (secondary), and θ_c
# -------------------------
n_vals = n_water_cauchy(lam_um, A, B, C)
r1 = r_internal_at_rainbow(n_vals, m=1)  # primary
r2 = r_internal_at_rainbow(n_vals, m=2)  # secondary
theta_c = critical_angle_inside(n_vals)

# Convert to degrees
r1_deg = np.degrees(r1)
r2_deg = np.degrees(r2)
tc_deg = np.degrees(theta_c)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Critical angle as a smooth black line
ax.plot(f_THz, tc_deg, lw=2.0, color='black', label='Critical angle')

# Color-coded scatter for primary and secondary using spectrum colors
for f, lam, y1, y2 in zip(f_THz, lam_nm, r1_deg, r2_deg):
    rgb = wavelength_to_rgb(lam)
    ax.scatter([f], [y1], s=18, color=rgb, edgecolor='none')
    ax.scatter([f], [y2], s=18, color=rgb, edgecolor='none')

# Overplot thin mean lines for clarity
ax.plot(f_THz, r1_deg, lw=1.0, color='red', label='Primary')
ax.plot(f_THz, r2_deg, lw=1.0, color='blue', label='Secondary')

# Axes, grid, labels
ax.set_title('Refraction angle of single and double rainbows', fontsize=14)
ax.set_xlabel('Frequency / THz')
ax.set_ylabel(r'$\phi$ / deg')  # here φ ≡ r (internal refraction angle)
ax.grid(True, alpha=0.3)


# Legend box
ax.legend(loc='lower right', frameon=True)



plt.tight_layout()
plt.show()
plt.savefig('11cimage.png')