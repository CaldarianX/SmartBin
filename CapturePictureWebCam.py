import cv2
from PIL import Image, ImageEnhance, ImageOps
import os
import random
import numpy as np


def augment_image(image):
    # Convert PIL image to numpy array for OpenCV
    image_np = np.array(image)
    
    # Random flip
    if random.choice([True, False]):
        image_np = cv2.flip(image_np, 1)
    
    # Random rotation
    angle = random.uniform(-30, 30)  # Random rotation between -30 to 30 degrees
    height, width = image_np.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    image_np = cv2.warpAffine(image_np, matrix, (width, height))
    
    # Convert back to PIL image for further augmentations
    image = Image.fromarray(image_np)

    # Random brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    return image

def save_frames_from_webcam(filename_prefix, save_directory, num_frames, file_extension='png'):
    # Open a connection to the webcam (0 is the default ID for the primary camera)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    frame_count = 0

    while frame_count < num_frames:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert the frame from BGR (OpenCV format) to RGB (Pillow format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create an Image object from the frame
        image = Image.fromarray(frame_rgb)
        
        # Augment the image
        augmented_image = augment_image(image)

        # Construct the full file path
        file_path = os.path.join(save_directory, f"{filename_prefix}_{frame_count + 1}.{file_extension}")

        # Save the augmented image
        augmented_image.save(file_path)
        print(f"Image saved at: {file_path}")

        frame_count += 1

    # Release the webcam
    cap.release()
    print(f"Saved {frame_count} frames.")


filename_prefix = "ME"
save_directory = "/Users/Luna/Desktop/ALL/Rain/MEEEE"
num_frames = 10
    
save_frames_from_webcam(filename_prefix, save_directory, num_frames, file_extension='png')
