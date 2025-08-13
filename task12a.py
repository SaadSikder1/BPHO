import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🔺 Triangular Prism White Light Dispersion (Incident from Left on AC Edge)")

# Prism parameters
prism_angle_deg = 60  # Equilateral triangle
prism_angle = np.radians(prism_angle_deg)

# Vertices of prism
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0.5, np.sin(prism_angle)])

# Refractive index for glass (simple Cauchy approx)
def refractive_index_glass(wavelength_nm):
    wl = wavelength_nm / 1000  # microns
    B = 1.5046
    C = 0.00420
    return B + C / (wl**2)

# Incident angle slider (degrees)
incident_angle_deg = st.slider("Incident angle of incoming ray (degrees from horizontal, 0=horizontal right)", 0, 60, 10)

incident_angle = np.radians(incident_angle_deg)

# AC edge vector
AC_vec = C - A

# Incident ray slope
m = np.tan(incident_angle)

# Find y0 for the incident ray to hit AC edge at t=0.5
t_intersect = 0.5
x_intersect = A[0] + t_intersect * AC_vec[0]
y_intersect = A[1] + t_intersect * AC_vec[1]

x0 = -1  # start x for incident ray
y0 = y_intersect - m * (x_intersect - x0)

P_in = np.array([x_intersect, y_intersect])
dir_in = np.array([np.cos(incident_angle), np.sin(incident_angle)])

# Normal to AC edge (outward)
edge_dir = AC_vec / np.linalg.norm(AC_vec)
normal_in = np.array([-edge_dir[1], edge_dir[0]])

# Ensure normal points outward
mid_AC = (A + C) / 2
vec_to_B = B - mid_AC
if np.dot(normal_in, vec_to_B) > 0:
    normal_in = -normal_in

# Angle of incidence
cos_theta_i = np.dot(-dir_in, normal_in)
theta_i = np.arccos(np.clip(cos_theta_i, -1, 1))

# Wavelengths for dispersion
wavelengths = np.linspace(380, 750, 100)  # nm
n_prism = refractive_index_glass(wavelengths)
n_air = 1.0

# Snell's law inside prism
sin_theta_r = n_air / n_prism * np.sin(theta_i)
theta_r = np.arcsin(np.clip(sin_theta_r, -1, 1))

# Refracted directions inside prism (rotated -normal by theta_r)
def rotate_vector(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ vec

dir_inside_all = np.array([rotate_vector(-normal_in, theta) for theta in theta_r])

# BC edge vector and normal (exit side)
BC_vec = C - B
normal_out = np.array([-BC_vec[1], BC_vec[0]])
mid_BC = (B + C) / 2
vec_to_A = A - mid_BC
if np.dot(normal_out, vec_to_A) > 0:
    normal_out = -normal_out

exit_points = []
exit_dirs = []

for i in range(len(wavelengths)):
    denom = dir_inside_all[i][0] * BC_vec[1] - dir_inside_all[i][1] * BC_vec[0]
    if abs(denom) < 1e-10:
        t = np.inf
    else:
        t = ((B[0] - P_in[0]) * BC_vec[1] - (B[1] - P_in[1]) * BC_vec[0]) / denom

    exit_point = P_in + t * dir_inside_all[i]

    proj = np.dot(exit_point - B, BC_vec) / np.dot(BC_vec, BC_vec)
    if 0 <= proj <= 1:
        exit_points.append(exit_point)
        exit_dirs.append(dir_inside_all[i])
    else:
        exit_points.append(None)
        exit_dirs.append(None)

# Calculate exit refraction angles (from prism to air)
theta_exit_i = np.array([np.arccos(np.clip(np.dot(-dir_inside_all[i], normal_out), -1, 1)) if exit_points[i] is not None else np.nan for i in range(len(wavelengths))])
sin_theta_exit_r = n_prism / n_air * np.sin(theta_exit_i)
theta_exit_r = np.arcsin(np.clip(sin_theta_exit_r, -1, 1))

# Refracted rays out of prism
dir_out_all = np.array([rotate_vector(-normal_out, theta) if not np.isnan(theta) else np.array([np.nan, np.nan]) for theta in theta_exit_r])

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Prism outline
prism_x = [A[0], B[0], C[0], A[0]]
prism_y = [A[1], B[1], C[1], A[1]]
ax.plot(prism_x, prism_y, 'k-', linewidth=2)

# Incident ray (yellow)
x_incident = np.linspace(x0 - 0.5, P_in[0], 100)
y_incident = y0 + m * (x_incident - x0)
ax.plot(x_incident, y_incident, color='yellow', linewidth=2, label="Incident ray")

# Normal lines (dashed) at incidence and exit
norm_len = 0.15
# At incidence
normal_line_in = np.array([P_in - normal_in * norm_len, P_in + normal_in * norm_len])
ax.plot(normal_line_in[:,0], normal_line_in[:,1], 'k--', linewidth=1, label='Normal at incidence')

# At exit (mean exit point to plot normal)
valid_exit_pts = [pt for pt in exit_points if pt is not None]
if valid_exit_pts:
    mean_exit = np.mean(valid_exit_pts, axis=0)
    normal_line_out = np.array([mean_exit - normal_out * norm_len, mean_exit + normal_out * norm_len])
    ax.plot(normal_line_out[:,0], normal_line_out[:,1], 'k--', linewidth=1, label='Normal at exit')

# Inside prism rays (rainbow colored)
for i, exit_pt in enumerate(exit_points):
    if exit_pt is not None:
        # Inside ray
        ax.plot([P_in[0], exit_pt[0]], [P_in[1], exit_pt[1]], color=plt.cm.plasma(i / len(exit_points)), linewidth=1)
        # Outgoing ray (rainbow)
        length = 0.5
        if not np.isnan(dir_out_all[i][0]):
            ax.plot([exit_pt[0], exit_pt[0] + dir_out_all[i][0]*length],
                    [exit_pt[1], exit_pt[1] + dir_out_all[i][1]*length],
                    color=plt.cm.plasma(i / len(exit_points)), linewidth=1)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.2, 1.2)
ax.set_aspect('equal')
ax.set_xlabel("X (arbitrary units)")
ax.set_ylabel("Y (arbitrary units)")
ax.set_title(f"Light through prism incident angle {incident_angle_deg}°")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close()
