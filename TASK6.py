import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Task 6", layout="centered")

st.title("Task 6")
st.write("Simulates real, inverted image formed by a thin lens.")

# Upload object image
uploaded_file = st.file_uploader("Upload an object image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.sidebar.header("Controls")

    # Lens focal length
    f = st.sidebar.slider("Focal length (f)", 0.1, 2.0, 1.0, 0.01)

    # Object position x > f
    x = st.sidebar.slider("Object x-position (x > f)", f + 0.1, 3.0, 2.0, 0.01)
    y = st.sidebar.slider("Object y-position", -1.0, 1.0, 0.0, 0.01)

    # Width of the image in plot
    obj_width = st.sidebar.slider("Image Width", 0.2, 1.0, 0.5, 0.01)
    obj_height = img_np.shape[0] / img_np.shape[1] * obj_width

    # Lens equation: X, Y for image
    X = -f * x / (x - f)
    Y = (y / x) * X

    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title("Image Formation by a Converging Lens")

    # Draw lens at x = 0
    lens_x = 0
    ax.axvline(x=lens_x, color='blue', linestyle='-', linewidth=2)
    ax.text(0, 1.8, "Lens", color='blue', ha='center')

    # Plot object
    ax.imshow(img_np, extent=[x, x + obj_width, y, y + obj_height])

    # Plot real inverted image (flipped vertically)
    flipped_img = np.flipud(img_np)
    ax.imshow(flipped_img, extent=[X - obj_width, X, Y - obj_height, Y])

    # Ray diagram (optional for clarity)
    ax.plot([x, 0, X], [y + obj_height / 2, 0, Y - obj_height / 2], 'r--', linewidth=1)

    st.pyplot(fig)

else:
    st.info("Upload an image to begin.")
