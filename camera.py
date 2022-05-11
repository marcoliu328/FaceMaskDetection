from tensorflow.keras.models import load_model
import imutils
import cv2

#load face_detector model (architecture definition and weights)
definitionPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceModel = cv2.dnn.readNet(definitionPath, weightsPath)

#load our mask_detector model from training
maskModel = load_model("face_mask_detection_model")

class Video(object):
    def __init__