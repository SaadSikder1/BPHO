import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# App title
st.title("Virtual Image in a Convex Spherical Mirror")

# Mirror characteristics
focal_length = -10  # convex mirror: focal length is negative
radius_of_curvature = 2 * focal_length

# User-controlled object distance
object_distance = st.slider(
    "Object Distance from Mirror (cm)", min_value=1, max_value=100, value=30.0, step=1.0
)

# Calculate image distance using mirror equation
image_distance = 1 / (1 / focal_length - 1 / object_distance)

# Define object height
object_height = 5

# Calculate image height using magnification formula
image_height = -image_distance / object_distance * object_height

# Set up plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-60, 60)
ax.set_ylim(-30, 30)
ax.set_aspect('equal')
ax.axis('off')

# Draw the convex mirror curve
x_mirror = np.linspace(-30, 30, 300)
y_mirror = -0.01 * x_mirror**2
ax.plot(x_mirror, y_mirror, color='black', linewidth=2)

# Principal axis
ax.plot([0, 0], [-25, 25], 'k--', linewidth=1)

# Focal point
ax.plot([-focal_length, -focal_length], [-1, 1], 'r')
ax.text(-focal_length, 2, 'F', color='r')

# Center of curvature
ax.plot([-radius_of_curvature, -radius_of_curvature], [-1, 1], 'b')
ax.text(-radius_of_curvature, 2, 'C', color='b')

# Draw object
ax.plot([object_distance, object_distance], [0, object_height], color='blue', linewidth=3)
ax.text(object_distance, object_height + 2, 'Object', ha='center')

# Draw image
ax.plot([image_distance, image_distance], [0, image_height], color='orange', linewidth=3)
ax.text(image_distance, image_height - 3, 'Image', ha='center')

# Ray 1: parallel to axis → appears to diverge from focal point
ax.plot([object_distance, 0], [object_height, object_height], 'gray')
ax.plot([0, image_distance], [object_height, image_height], 'gray', linestyle='dashed')

# Ray 2: toward mirror center → reflects back
ax.plot([object_distance, 0], [object_height, 0], 'gray')
ax.plot([0, image_distance], [0, image_height], 'gray', linestyle='dashed')

# Show plot
st.pyplot(fig)
