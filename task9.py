import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage import affine_transform

# Page config
st.set_page_config(page_title="Convex Mirror Simulator", layout="centered")
st.title("🔍 Convex Mirror Image Simulation")
st.markdown("""
This interactive model shows the virtual image formed by a **convex spherical mirror**.  
Upload an object image and adjust parameters to explore how the image changes.
""")

# Upload image
uploaded_file = st.file_uploader("Upload an image to reflect", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and convert image
    img = Image.open(uploaded_file).convert("L")
    img_np = np.array(img)

    # Sidebar controls
    st.sidebar.header("Mirror Parameters")
    focal_length = st.sidebar.slider("Focal Length (+, units)", 1.0, 20.0, 5.0, 0.5)
    object_distance = st.sidebar.slider("Object Distance (units)", 1.0, 50.0, 20.0, 0.5)

    # Mirror equation for convex: 1/f = 1/do + 1/di  →  di = 1 / (1/f - 1/do)
    f = focal_length
    do = object_distance
    try:
        di = 1 / (1/f + 1/do)
    except ZeroDivisionError:
        di = np.inf

    magnification = abs(di / do)
    image_size = max(10, int(magnification * img_np.shape[0]))

    # Resize and mirror
    img_virtual = Image.fromarray(img_np)
    img_virtual = img_virtual.resize((image_size, image_size))
    img_virtual = ImageOps.mirror(img_virtual)

    # Prepare plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img_np, cmap='gray')
    ax[0].set_title("Original Object")
    ax[1].imshow(np.array(img_virtual), cmap='gray')
    ax[1].set_title("Virtual Image (Convex Mirror)")

    for a in ax:
        a.axis('off')
    st.pyplot(fig)

    st.markdown(f"""
    **Mirror Equation:** 1/f = 1/do + 1/di  
    **Image Distance:** {di:.2f} units (virtual)  
    **Magnification:** {magnification:.2f}  
    """)
else:
    st.info("Upload an image above to begin the simulation.")
