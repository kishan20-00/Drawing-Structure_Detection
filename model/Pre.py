import torch
import torchvision
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load a pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def extract_structure(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Transform the image to a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Get predictions from the model
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Visualize the results
    draw = ImageDraw.Draw(image)
    for element in range(len(prediction[0]['masks'])):
        mask = prediction[0]['masks'][element, 0]
        mask = mask.mul(255).byte().cpu().numpy()
        
        # Threshold the mask to get binary regions
        thresholded_mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        # Draw the detected region
        contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            draw.polygon(contour.reshape(-1, 2), outline="red", fill=None)
    
    # Show the result
    plt.imshow(image)
    plt.title("Detected Structures")
    plt.show()

# Test with an image path
image_path = 'Notes1.jpg'
extract_structure(image_path)
