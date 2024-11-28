import cv2
import numpy as np
from PIL import Image

def extract_diagrams(image_path, output_path):
    """
    Automatically detects and highlights diagrams in an image.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the highlighted image.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and draw bounding boxes for large regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small regions (adjust thresholds as needed)
        if w > 50 and h > 50:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Highlighted image saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "Notes1.jpg"
    output_image_path = "path_to_highlighted_image.jpg"
    extract_diagrams(input_image_path, output_image_path)
