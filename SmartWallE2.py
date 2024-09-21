import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite
import os
import time
import pigpio

class_names = {
    0: "BlueBottle",
    1: "GreenBottle",
    2: "OrangeBottle",
    3: "MilkBox",
    4: "BrownCan",
    5: "RedCan",
    6: "ketchup",
    7: "Empty",
}




pi = pigpio.pi()
################################
# Edit here
# GPIO pin for the 180-degree servo
servo_180_gpio = "YOURPIN"
# GPIO pin for the 360-degree servo
servo_360_gpio = "YOURPIN"
################################
# Set initial positions
pi.set_servo_pulsewidth(servo_180_gpio, 1500)  # Neutral position for 180-degree servo
pi.set_servo_pulsewidth(servo_360_gpio, 1500)  # Neutral position for 360-degree servo

################################
# USEFULL FUNCTION
def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def preprocess_image(frame,input_shape):
    img = cv.flip(frame, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Ensure RGB format
    img = cv.resize(img, (input_shape[1], input_shape[2]))  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def openBoard():
    pi.set_servo_pulsewidth(servo_180_gpio, 500)
    time.sleep(1)
    
    # Move to 90 degrees
    pi.set_servo_pulsewidth(servo_180_gpio, 1500)
    time.sleep(1)
    
    # Move to 180 degrees
    pi.set_servo_pulsewidth(servo_180_gpio, 2500)
    time.sleep(1)

def closeBoard():
    pi.set_servo_pulsewidth(servo_180_gpio, 2500)
    time.sleep(1)
    
    # Move to 90 degrees
    pi.set_servo_pulsewidth(servo_180_gpio, 1500)
    time.sleep(1)
    
    # Move to 180 degrees
    pi.set_servo_pulsewidth(servo_180_gpio, 500)
    time.sleep(1)

def rotateBoard(t):
    pi.set_servo_pulsewidth(servo_360_gpio, 1000)
    time.sleep(t)
    pi.set_servo_pulsewidth(servo_360_gpio, 1500)

def rotateBoardBack(t):
    pi.set_servo_pulsewidth(servo_360_gpio, 2500)
    time.sleep(t)
    pi.set_servo_pulsewidth(servo_360_gpio, 1500)

def Wall_E(t):
    rotateBoard(t)
    openBoard()
    closeBoard()
    rotateBoardBack(t)

################################
# Config here ! ! !
model_path = '/home/projectemeryl/Documents/Sequences/TFlite/model_unquant.tflite'
################################

################################
# Main Code Don't Touch!!!!
interpreter = load_model(model_path)
input_shape = interpreter.get_input_details()[0]['shape']

cap = cv.VideoCapture(0)

while True:
    ret,image = cap.read() #read frame from webcam ret = success?

    frame = preprocess_image(image,input_shape)
    predictions = predict_image(interpreter, frame)

    predicted_class = int(np.argmax(predictions)) #find index with max value in predicted
    predicted_probability = np.max(predictions) #get the max value
    classname = class_names[predicted_class]
    print("This is",classname)
    if predicted_class == 0 and predicted_class == 1 and predicted_class == 2 :
        print("Bottle")
        Wall_E(1) #Tune here  90
    elif predicted_class == 3:
        print("Box")
        Wall_E(2) #Tune here 180
    elif predicted_class == 4 and predicted_class == 5:
        print("Can")
        Wall_E(3) #Tune here 270
    elif predicted_class == 6:
        print("Glass")
        Wall_E(4) #Tune here 360
    else:
        print("Error??? WTF predicted class is",predicted_class)
        print("Tell prom Asap")
    cv.putText(image,str(classname) + " " + str(float("%.2f"%predicted_probability)),(10,80),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),7)
    cv.imshow("Frame",image) #show frame on computer
    cv.waitKey(1)
    print("End of one Frame")