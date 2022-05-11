from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict(frame, faceNet, maskNet):

    #grab dimensions of frame and make a blob -> to be put into face detector
    #performs mean subtraction and resizes to expected dnn input size (in this case 224x224)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    #send the blob to the nn and grab face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    #initialization
    faces = []
    locs = []
    preds = []

    #iterate over detections
    for i in range(detections.shape[2]):
        #grab probability of detection
        confidence = detections[0 , 0, i, 2]

        #filter out weak detections
        if confidence > 0.7:
            #compute coordinates for the bounding box
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #make sure the bounding box is within the image
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #extract the face ROI, now we have an image (array) ready for mask detection
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #append to lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    #now run through each face into the mask detector
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    #return the location of the face, and prediction (percentage)
    return (locs, preds)

"""
#load face_detector model (architecture definition and weights)
definitionPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceModel = cv2.dnn.readNet(definitionPath, weightsPath)

#load our mask_detector model from training
maskModel = load_model("face_mask_detection_model")

#start the video stream
vs = VideoStream(src=0, resolution=(640,480)).start()


#loop over frames of videoStream and evaluate
while True:
    
    #grab frame from video and resize to width of 800p
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    #send frame and both models for prediction
    (locs,preds) = detect_and_predict(frame, faceModel, maskModel)

    #analyze detected face locations
    for (box, pred) in zip(locs, preds):

        #grab coordinates and prediction, set label and colour (BGR)
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred  

        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0) #Green
        else:
            label = "No Mask"
            color = (0, 0, 255) #Red

        #Add the probability of label to the label (floating point)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #output the label and bounding box
        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness=2)

    #show the frame
    cv2.imshow("Frame", frame)

    #quit when esc is pressed
    key = cv2.waitKey(1) & 0xFF #bitwise AND to leave last 8 bits of waitKey
    if key ==  27: #the esc key
        break
        
#clean up
cv2.destroyAllWindows()
vs.stop()
"""