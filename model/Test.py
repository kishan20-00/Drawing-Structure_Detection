import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'Notes1.jpg'  # Replace with the actual image path
image = cv2.imread(image_path)

# Convert the original image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to emphasize the drawings (ignoring text areas)
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to clean up small noise (which could be text)
kernel = np.ones((5, 5), np.uint8)
cleaned_image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

# Find contours in the cleaned image (drawings)
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw the contours on
final_output_image = image.copy()

# Draw contours only on larger regions that are likely to be diagrams (drawings)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # A threshold for the size of the drawn regions (adjust as needed)
        cv2.drawContours(final_output_image, [contour], -1, (0, 255, 0), 2)  # Green color for contours

# Invert the adaptive thresholded image to get the unmarked (text/other) areas
inverted_mask = cv2.bitwise_not(adaptive_thresh)

# Find contours in the inverted mask (unmarked areas)
unmarked_contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around unmarked features
for contour in unmarked_contours:
    area = cv2.contourArea(contour)
    if area > 500:  # Adjust this threshold for the size of unmarked regions
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a red bounding box around unmarked features
        cv2.rectangle(final_output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save the final output image
output_path = 'final_output_with_features.jpg'
cv2.imwrite(output_path, final_output_image)

# Display the final output
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(final_output_image, cv2.COLOR_BGR2RGB))
plt.title("Final Output: Drawings and Unmarked Features")
plt.axis('off')  # Remove axes for better visualization
plt.show()

# Optionally, notify the user about the saved image
print(f"Final output saved at: {output_path}")
