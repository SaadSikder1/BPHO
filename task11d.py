import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("🌈 Rainbow Visibility vs Solar Elevation")

# Constants: typical rainbow angular radii (degrees)
R_PRIMARY = 42
R_SECONDARY = 51

def plot_rainbows(solar_elev_deg):
    center_elev = -solar_elev_deg  # antisolar point elevation

    theta = np.linspace(0, 2*np.pi, 1000)

    def rainbow_arc(radius_deg):
        x = radius_deg * np.cos(theta)
        y = radius_deg * np.sin(theta) + center_elev
        return x, y

    x1, y1 = rainbow_arc(R_PRIMARY)
    x2, y2 = rainbow_arc(R_SECONDARY)

    # Keep only points above horizon
    mask1 = y1 >= 0
    mask2 = y2 >= 0

    plt.figure(figsize=(6,6))
    plt.plot(x1[mask1], y1[mask1], color='red', linewidth=3, label='Primary Rainbow')
    plt.plot(x2[mask2], y2[mask2], color='blue', linewidth=3, label='Secondary Rainbow')

    plt.axhline(0, color='gray', linestyle='--')  # horizon
    plt.title(f"Solar Elevation: {solar_elev_deg:.1f}°")
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Elevation (degrees)")
    plt.xlim(-60, 60)
    plt.ylim(0, 60)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(plt)
    plt.close()

solar_elevation = st.slider("Solar Elevation Angle (degrees)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)

plot_rainbows(solar_elevation)
