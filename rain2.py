import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite
import os

class_names = {
    0: "BlueBottle",
    1: "GreenBottle",
    2: "OrangeBottle",
    3: "MilkBox",
    4: "BrownCan",
    5: "RedCan",
    6: "ketchup",
}

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



################################
# Config here ! ! !
model_path = 'TO/YOUR/Model'
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

    predicted_class = np.argmax(predictions) #find index with max value in predicted
    predicted_probability = np.max(predictions) #get the max value
    classname = class_names[predicted_class]

    cv.putText(image,str(classname) + " " + str(float("%.2f"%predicted_probability)),(50,150),cv.FONT_HERSHEY_SIMPLEX,4,(0,0,255),10)
    cv.imshow("Frame",image) #show frame on computer
    cv.waitKey(1)
################################