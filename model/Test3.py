import streamlit as st
import cv2
import numpy as np
from PIL import Image

def extract_diagrams_and_boxes(image):
    """
    Detects diagrams in an image and extracts individual bounding boxes.

    Args:
        image (numpy array): Input image as a NumPy array.

    Returns:
        numpy array: Image with highlighted diagrams.
        list of numpy arrays: Cropped regions of detected diagrams.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list for cropped boxes
    cropped_boxes = []

    # Loop through contours and draw bounding boxes for large regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small regions (adjust thresholds as needed)
        if w > 50 and h > 50:
            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Crop the bounding box from the original image
            cropped_box = image[y:y+h, x:x+w]
            cropped_boxes.append(cropped_box)

    return image, cropped_boxes

# Streamlit App
st.title("Diagram Detection App")
st.write("Upload an image, and the app will detect diagrams and show each box separately.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Extract diagrams and individual boxes
    highlighted_image, cropped_boxes = extract_diagrams_and_boxes(image_np)

    # Convert highlighted image to PIL for display
    highlighted_pil = Image.fromarray(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))

    # Display the highlighted image
    st.image(image, caption="Original Image", use_container_width=True)
    st.image(highlighted_pil, caption="Highlighted Image with Boxes", use_container_width=True)

    # Show each box separately
    st.write("### Individual Diagrams")
    if cropped_boxes:
        for i, box in enumerate(cropped_boxes):
            # Convert each cropped box to PIL format for display
            box_pil = Image.fromarray(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
            st.image(box_pil, caption=f"Diagram {i+1}", use_container_width=False)
    else:
        st.write("No diagrams detected!")

    # Option to download the highlighted image
    st.write("### Download the Highlighted Image")
    result_path = "highlighted_diagram.png"
    highlighted_pil.save(result_path)
    with open(result_path, "rb") as file:
        btn = st.download_button(
            label="Download Highlighted Image",
            data=file,
            file_name="highlighted_diagram.png",
            mime="image/png"
        )
