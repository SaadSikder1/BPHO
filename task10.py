import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Cylindrical Anamorphic Projection", layout="centered")
st.title("Cylindrical Anamorphic Image Generator")

# Core conversion function (θ and R to image coordinates)
def convt(R, b, cols, c, Rmin, Rmax):
    norm_r = (R - Rmin) / (Rmax - Rmin)
    x = int(b * cols / (2 * math.pi)) % cols
    y = int(c - norm_r * c)
    return x, y

# Main warping function
def generate_anamorphic_image(img_np, output_size):
    rows, cols = output_size, output_size
    c = rows // 2
    rmin = 0.3 * c  # inner radius (where cylinder sits)
    rmax = c        # outer edge

    # White background instead of black
    warp = np.full((rows, cols, 3), 255, dtype=np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            dx = i - c
            dy = c - j  # flip y
            R = math.sqrt(dx*dx + dy*dy)

            if rmin <= R <= rmax:
                b = math.atan2(dy, dx)
                if b < 0:
                    b += 2 * math.pi
                x, y = convt(R, b, img_np.shape[1], c, rmin, rmax)

                # Clamp
                x = np.clip(x, 0, img_np.shape[1] - 1)
                y = np.clip(y, 0, img_np.shape[0] - 1)

                warp[j, i] = img_np[y, x]

    return warp

# Upload image
uploaded_file = st.file_uploader("Upload a square image (will wrap into circle)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((512, 512))  # enforce square
    img_np = np.array(img)

    st.sidebar.header("Output Settings")
    output_size = st.sidebar.slider("Output Size", 300, 1000, 600, 50)

    warped = generate_anamorphic_image(img_np, output_size)

    fig, ax = plt.subplots()
    ax.imshow(warped)
    ax.set_title("Wrapped Image for Cylindrical Mirror (White Background)")
    ax.axis('off')
    st.pyplot(fig)

    st.markdown("Place a polished **cylinder** in the center to reveal the original image.")
else:
    st.info("Please upload a square image.")
