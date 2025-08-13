import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="BK7 Prism Dispersion", layout="centered")

st.title("Dynamic Model: White-Light Dispersion by a BK7 Triangular Prism")

st.markdown("""This app uses the **Sellmeier equation** for BK7 crown glass and **Snell's law**
to compute the angular deviation and draw a schematic prism with dispersive rays.If nothing shows on the graph, that must mean there is something wrong with your inputs.
""")

# Dispersion model
def n_BK7(lambda_nm):
    x = np.array(lambda_nm, dtype=float) / 1000.0
    a = np.array([1.03961212, 0.231792344, 1.01146945])
    b = np.array([0.00600069867, 0.0200179144, 103.560653])
    y = np.zeros_like(x, dtype=float)
    for ak, bk in zip(a, b):
        y += (ak * x**2) / (x**2 - bk)
    return np.sqrt(1.0 + y)

def wl_to_rgb(l):
    if l < 405: return (0.0,0.0,0.0)
    if l < 480: return (1.0,0.0,0.0)             # Red
    if l < 510: return (1.0,127/255.0,0.0)       # Orange
    if l < 530: return (1.0,1.0,0.0)             # Yellow
    if l < 600: return (0.0,1.0,0.0)             # Green
    if l < 620: return (0.0,1.0,1.0)             # Cyan
    if l < 680: return (0.0,0.0,1.0)             # Blue
    if l < 790: return (127/255.0,0.0,1.0)       # Violet
    return (0.0,0.0,0.0)

def prism_deviation(theta_i_deg, A_deg, lambda_nm):
    n = float(n_BK7([lambda_nm])[0])
    ti = np.deg2rad(theta_i_deg)
    A  = np.deg2rad(A_deg)
    r1 = np.arcsin(np.sin(ti)/n)
    r2 = A - r1
    arg = n*np.sin(r2)
    if np.abs(arg) > 1.0:
        return np.nan, np.nan, np.rad2deg(r1), np.rad2deg(r2), True
    theta_t = np.arcsin(arg)
    delta = ti + theta_t - A
    return np.rad2deg(delta), np.rad2deg(theta_t), np.rad2deg(r1), np.rad2deg(r2), False

# Controls
cols = st.columns(3)
with cols[0]:
    theta_i = st.slider("Incident angle θᵢ (deg)", 0.0, 30.0, 7.0, 0.1)
with cols[1]:
    A = st.slider("Apex angle α (deg)", 10.0, 80.0, 45.0, 0.1)
with cols[2]:
    N = st.slider("Number of rays", 10, 120, 60, 5)

wavelengths = np.linspace(405, 790, N)
res = np.array([prism_deviation(theta_i, A, wl) for wl in wavelengths])
deltas = res[:,0]
TIR = np.isnan(deltas)

# Plot: δ(λ)
fig1 = plt.figure(figsize=(6,4))
plt.plot(wavelengths[~TIR], deltas[~TIR], linewidth=2)
plt.xlabel("Wavelength λ (nm)")
plt.ylabel("Total deviation δ (degrees)")
plt.title("Angular dispersion δ(λ)")
plt.grid(True)
st.pyplot(fig1)

# Plot: prism schematic with exiting rays
fig2 = plt.figure(figsize=(6,4))
base = 3.5
tri_x = np.array([0.0,  base*np.cos(np.deg2rad(A)),  base])
tri_y = np.array([0.0,  base*np.sin(np.deg2rad(A)), 0.0])
plt.plot([tri_x[0], tri_x[1], tri_x[2], tri_x[0]], [tri_y[0], tri_y[1], tri_y[2], tri_y[0]], 'k-', linewidth=1)
x0, y0 = 0.0, 2.0*0.6
ti_rad = np.deg2rad(theta_i)
xi = np.linspace(-1.5, x0, 50)
yi = y0 + np.tan(-ti_rad)*(xi - x0)
plt.plot(xi, yi, linestyle='--', linewidth=1)

x_exit_start, y_exit_start = base*np.cos(np.deg2rad(A))*0.9, base*np.sin(np.deg2rad(A))*0.9
for wl in wavelengths:
    delta_deg, theta_t_deg, r1_deg, r2_deg, is_TIR = prism_deviation(theta_i, A, wl)
    if is_TIR: 
        continue
    normal2_angle = -np.deg2rad(A)
    exit_angle_world = normal2_angle - np.deg2rad(theta_t_deg)
    L = 3.5
    xs = np.linspace(x_exit_start, x_exit_start + L*np.cos(exit_angle_world), 2)
    ys = np.linspace(y_exit_start, y_exit_start + L*np.sin(exit_angle_world), 2)
    plt.plot(xs, ys, linewidth=2, color=wl_to_rgb(wl))

plt.title("Prism schematic (exiting rays colored by λ)")
plt.xlabel("x (arb. units)")
plt.ylabel("y (arb. units)")
plt.axis("equal")
plt.grid(True)
st.pyplot(fig2)

st.caption("Physics: Sellmeier dispersion for BK7; Snell's law at both faces; δ = θᵢ + θₜ − α.")