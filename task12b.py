import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import gridspec

# Sellmeier equation for BK7
def n_BK7(lambda_nm):
    lam_um = lambda_nm / 1000.0
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
    return np.sqrt(1 + (B1*lam_um**2)/(lam_um**2 - C1) +
                      (B2*lam_um**2)/(lam_um**2 - C2) +
                      (B3*lam_um**2)/(lam_um**2 - C3))

# Conversion wavelength → RGB
def wl_to_rgb(wl):
    if wl < 380 or wl > 780: return (0,0,0)
    if wl < 440: r, g, b = -(wl-440)/(440-380), 0, 1
    elif wl < 490: r, g, b = 0, (wl-440)/(490-440), 1
    elif wl < 510: r, g, b = 0, 1, -(wl-510)/(510-490)
    elif wl < 580: r, g, b = (wl-510)/(580-510), 1, 0
    elif wl < 645: r, g, b = 1, -(wl-645)/(645-580), 0
    else: r, g, b = 1, 0, 0
    return (max(0,min(1,r)), max(0,min(1,g)), max(0,min(1,b)))

# Prism calculation
def prism_angles(theta_i_deg, A_deg, lambda_nm):
    n = n_BK7(lambda_nm)
    ti = np.deg2rad(theta_i_deg)
    A = np.deg2rad(A_deg)
    r1 = np.arcsin(np.sin(ti)/n)
    r2 = A - r1
    arg = n*np.sin(r2)
    if np.abs(arg) > 1:
        return np.nan, np.nan
    theta_t = np.arcsin(arg)
    delta = np.rad2deg(ti + theta_t - A)
    return np.rad2deg(theta_t), delta

# Parameters
alpha = 45
lambda_nm = 553  # green reference wavelength
f_THz = 3e5 / lambda_nm  # approx frequency in THz
theta_i_vals = np.linspace(0.1, 80, 500)

theta_t_vals, delta_vals = [], []
for ti in theta_i_vals:
    tt, d = prism_angles(ti, alpha, lambda_nm)
    theta_t_vals.append(tt)
    delta_vals.append(d)

theta_t_vals = np.array(theta_t_vals)
delta_vals = np.array(delta_vals)

# Multi-alpha plot (bottom-right)
alphas = np.arange(10, 85, 5)
delta_map = []
for A in alphas:
    row = []
    for ti in theta_i_vals:
        _, d = prism_angles(ti, A, lambda_nm)
        row.append(d)
    delta_map.append(row)

# Start figure
fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[1,1])
gs.update(wspace=0.4, hspace=0.4)

# Top-left plot: Transmission angle vs incidence
ax1 = plt.subplot(gs[0,0])
ax1.plot(theta_i_vals, theta_t_vals, 'b')
ax1.set_xlabel(r"Angle of incidence $\theta_i$ / deg")
ax1.set_ylabel(r"Transmission angle $\theta_t$ / deg")
ax1.set_title(r"$\theta_t$ vs $\theta_i$, $\alpha=45^\circ$, $\lambda={}nm$".format(lambda_nm))
eq_text = r"$\sin\theta_t = \sqrt{n^2 - \sin^2\theta_i}\sin\alpha - \sin\theta_i\cos\alpha$"
ax1.text(0.5, 0.85, eq_text, fontsize=10, transform=ax1.transAxes,
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))

# Middle plot: Prism schematic
ax2 = plt.subplot(gs[0,2])
A_rad = np.deg2rad(alpha)
prism_poly = Polygon([[0,0],[np.cos(A_rad), np.sin(A_rad)],[1.5,0]],
                     closed=True, fill=False, color='white', linewidth=1)
ax2.add_patch(prism_poly)
# Rays
wls = np.linspace(420, 680, 15)
for wl in wls:
    _, d = prism_angles(7, alpha, wl)
    if np.isnan(d): continue
    exit_angle = np.deg2rad(d)
    ax2.plot([np.cos(A_rad), np.cos(A_rad)+0.8*np.cos(exit_angle)],
             [np.sin(A_rad), np.sin(A_rad)+0.8*np.sin(exit_angle)],
             color=wl_to_rgb(wl), linewidth=2)
ax2.set_facecolor('black')
ax2.set_aspect('equal')
ax2.set_title(r"$\theta_i = 7^\circ, \alpha = 45^\circ$")


# Bottom-left plot: Deflection vs incidence
ax4 = plt.subplot(gs[1,0])
ax4.plot(theta_i_vals, delta_vals, 'b')
ax4.set_xlabel(r"Angle of incidence $\theta_i$ / deg")
ax4.set_ylabel(r"Deflection angle $\delta$ / deg")
ax4.set_title(r"$\delta$ vs $\theta_i$, $\alpha=45^\circ$")
ax4.text(0.5, 0.85, r"$\delta = \theta_i + \theta_t - \alpha$", fontsize=10,
         transform=ax4.transAxes,
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))

# Bottom-right plot: Deflection contour for multiple α
ax5 = plt.subplot(gs[1,1])
for i, A in enumerate(alphas):
    ax5.plot(theta_i_vals, delta_map[i], label=r"$\alpha={}$".format(A))
ax5.set_xlabel(r"Angle of incidence $\theta_i$ / deg")
ax5.set_ylabel(r"Deflection angle $\delta$ / deg")
ax5.set_title(r"$\delta$ for various $\alpha$")
ax5.legend(fontsize=8)

# Hide bottom-right empty cell
ax_empty = plt.subplot(gs[1,2])
ax_empty.axis('off')

plt.show()

plt.savefig('12bimage.png')