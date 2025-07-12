import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from PIL import Image

# Page settings
st.set_page_config(page_title="Task 8 - Concave Mirror", layout="centered")
st.title("Task 8: Concave Mirror Image Simulation")
st.markdown("""
Simulate how an image appears when reflected through a **concave mirror**.
Upload any image and adjust the parameters to see the transformation.
""")

# Upload section
uploaded_file = st.file_uploader("Upload an image to reflect", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and convert image to grayscale for simplicity
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)

    # Sidebar controls
    st.sidebar.header("Mirror Parameters")
    angle = st.sidebar.slider("Reflection Angle (°)", 0, 360, 180)
    scale = st.sidebar.slider("Reflection Scale (zoom)", 0.1, 2.0, 0.5, 0.01)

    # Create transformation matrix (rotate + scale)
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    transform_matrix = np.array([
        [scale * cos_theta, -scale * sin_theta],
        [scale * sin_theta,  scale * cos_theta]
    ])

    # Offset to keep image centered during transformation
    center = np.array(img_np.shape) / 2
    offset = center - transform_matrix @ center

    # Apply transformation
    warped = affine_transform(img_np, transform_matrix, offset=offset, output_shape=img_np.shape)

    # Display results
    st.markdown("### Comparison")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img_np, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(warped, cmap='gray')
    ax[1].set_title("Reflected Image (Concave Mirror)")
    for a in ax:
        a.axis('off')
    st.pyplot(fig)
else:
    st.info("📤 Upload an image to begin the simulation.")
