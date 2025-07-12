import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

st.set_page_config(page_title="Task 6 and 7", layout="centered")
st.title("Task 6 and Task 7")

# Upload image
uploaded_file = st.file_uploader("Upload an object image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Sidebar controls
    st.sidebar.header("Controls")
    f = st.sidebar.slider("Focal length (f)", 0.1, 2.0, 1.0, 0.01)
    x = st.sidebar.slider("Object x-position", 0.1, 3.0, 0.8, 0.01)
    y = st.sidebar.slider("Object y-position", -1.5, 1.5, 0.0, 0.01)
    obj_width = st.sidebar.slider("Image Width", 0.2, 1.0, 0.5, 0.01)
    obj_height = img_np.shape[0] / img_np.shape[1] * obj_width

    # Setup figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Lens Image Formation")

    ray_color = 'magenta'
    lens_color = (0.7, 0.85, 1.0, 0.5)

    # Calculate image position
    X, Y = None, None
    image_extent = None
    is_real = False

    if x > f:
        # Real image (inverted)
        X = -f * x / (x - f)
        Y = (y / x) * X
        flipped_img = np.flipud(img_np)
        image_extent = [X - obj_width, X, Y - obj_height, Y]
        is_real = True
    else:
        # Virtual image (upright)
        X = f * x / (x - f)
        Y = (y / x) * X
        image_extent = [X, X + obj_width, Y, Y + obj_height]

    # Dynamic plot limits
    all_x = [x, x + obj_width, image_extent[0], image_extent[1]]
    all_y = [y, y + obj_height, image_extent[2], image_extent[3]]
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw axis and lens
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    lens = Ellipse((0, 0), width=0.3, height=3.5, facecolor=lens_color, edgecolor='blue', lw=1.5)
    ax.add_patch(lens)
    ax.axvline(x=0, color='blue', linestyle='-', linewidth=2)
    ax.text(0, max(all_y) - 0.2, "Lens", color='blue', ha='center')

    # Focal points
    ax.plot([f, -f], [0, 0], 'b*', markersize=10)
    ax.text(f, -0.2, "F", ha='center', color='blue')
    ax.text(-f, -0.2, "F", ha='center', color='blue')

    # Draw object
    ax.imshow(img_np, extent=[x, x + obj_width, y, y + obj_height])

    # Draw image and rays
    if is_real:
        ax.imshow(flipped_img, extent=image_extent)
        ax.text(X - 0.2, image_extent[2] - 0.2, "Real Image", color='green')
        ax.plot([x, 0, X], [y + obj_height / 2, 0, image_extent[2] + obj_height / 2], ray_color, linestyle='--')
    else:
        ax.imshow(img_np, extent=image_extent)
        ax.text(X + 0.1, image_extent[3] + 0.1, "Virtual Image", color='orange')
        ax.plot([x + obj_width, 0], [y + obj_height / 2, 0], ray_color, linestyle='--')
        ax.plot([0, X], [0, image_extent[2] + obj_height / 2], ray_color, linestyle='--')

    st.pyplot(fig)
else:
    st.info("Please upload an object image.")
#streamlit run /workspaces/BPHO/TASK6AND7.py
# THIS IS THE COMMAND YOU USE
