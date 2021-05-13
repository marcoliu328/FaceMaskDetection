from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INITIAL_LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 32

DIRECTORY = r'/Users/marcoliu/Desktop/Github/FaceMaskDetection/dataset'
print(DIRECTORY)
RESULTS = ["Masked", "Unmasked"]

print("--- Loading images into data structures ---")

data = [] #the image dataset
labels = [] #the ground truth values (masked or unmasked)

#PREPROCESS IMAGES
#load each image, convert to 224x224 for MobileNet, convert from PIL image to numpy array, preprocess input
for result in RESULTS:
    path = os.path.join(DIRECTORY, result) 
    for img in os.listdir(path): 
        img_path = os.path.join(path, img) #find individual image paths
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image) #required for MobileNet

        #add to data structures for learning
        data.append(image)
        labels.append(result)

#use one-hot encoding for the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels) 
labels = to_categorical(labels) #converts integer vector into binary class matrix

#convert data and labels into numpy arrays
data = np.array(data, dtype="float32")


#split data into test and training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state = 1)


