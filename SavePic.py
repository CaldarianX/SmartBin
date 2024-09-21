import cv2
from PIL import Image
import os
import time

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

        # Construct the full file path
        file_path = os.path.join(save_directory, f"{filename_prefix}_{frame_count + 1}.{file_extension}")

        # Convert the frame from BGR (OpenCV format) to RGB (Pillow format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb,(224,224))
        # Create an Image object from the frame
        image = Image.fromarray(frame_rgb)
        
        # Save the image
        image.save(file_path)
        print(f"Image saved at: {file_path}")

        frame_count += 1

    # Release the webcam
    cap.release()
    print(f"Saved {frame_count} frames.")

def Wait():
    print("Start in 3")
    time.sleep(1)
    print("Start in 2")
    time.sleep(1)
    print("Start in 1")
    time.sleep(1)
    print("Start!")
    
####################################################################
filename_prefix = "RedBottle" #Red bottle?
save_directory = "Bottle" #Bottle Box Can Glass?
num_frames = 500
####################################################################

Wait()
save_directory = "./PictureData/" + save_directory
save_frames_from_webcam(filename_prefix, save_directory, num_frames, file_extension='png')
