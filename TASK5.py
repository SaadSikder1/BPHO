import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plane Mirror Reflection", layout="centered")

st.title("Reflection in a Plane Mirror")

# Upload image
uploaded_file = st.file_uploader("Upload an image to reflect (the 'object')", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.sidebar.header("Object Position Controls")
    x_offset = st.sidebar.slider("X Position", -1.0, 1.0, 0.2, 0.01)
    y_offset = st.sidebar.slider("Y Position", -0.5, 0.5, 0.0, 0.01)
    img_width = st.sidebar.slider("Image Width", 0.1, 1.0, 0.4, 0.01)

    img_height = img_np.shape[0] / img_np.shape[1] * img_width

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.set_title("Reflection in a Plane Mirror")

    # Draw mirror
    ax.axvline(x=0, color='blue', linestyle='-', linewidth=1, label="Mirror")

    # Draw object and virtual image
    ax.imshow(img_np, extent=[x_offset, x_offset + img_width, y_offset, y_offset + img_height])
    ax.imshow(np.fliplr(img_np), extent=[-x_offset - img_width, -x_offset, y_offset, y_offset + img_height], alpha=0.7)

    ax.legend(["Mirror"])
    st.pyplot(fig)
else:
    st.info("Upload an image to begin.")