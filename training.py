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


#increase the data set by using ImageDataGenerator for data augmentation
augment = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,fill_mode="nearest")


#import the base model MobileNetV2 while excluding the top fully connected layers to prepare for fine tuning
#images are of shape 224px224x3 (RGB)
mobile = MobileNetV2(include_top=False, input_tensor=Input(shape=(224,224,3)))


#construct the rest of model, this is a functional model, not sequential
#second paramter is to indicate the "previous" parts of the model
top = mobile.output
top = AveragePooling2D(pool_size=(7,7))(top) #add pooling layer
top = Flatten(name="Flatten")(top) #flatten for fc
top = Dense(128, activation="relu")(top) #use relu for non-linear use cases
top = Dropout(0.5)(top) #drops some neurons at random to prevent overfitting
top = Dense(2, activation="softmax")(top) #can use sigmoid as well since we have only 2 output classes


#construct the full model
model = Model(inputs=mobile.input, outputs=top)


#freeze the layers from mobilenets since we dont need to train them
for layer in mobile.layers:
    layer.trainable = False


#compile model
opt = Adam(lr=INITIAL_LEARNING_RATE, decay=INITIAL_LEARNING_RATE/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


#train the head of the model
H = model.fit(augment.flow(trainX, trainY, batch_size=BATCH_SIZE), steps_per_epoch=len(trainX) // BATCH_SIZE, validation_data=(testX, testY), validation_steps=len(testX) // BATCH_SIZE, epochs=EPOCHS)


#make predictions
predictions = model.predict(testX, batch_size=BATCH_SIZE)
predictions = np.argmax(predictions, axis = 1)


#print training report
print(classification_report(y_true=testY.argmax(axis=1), y_pred=predictions, target_names=lb.classes_))


#save the model
model.save("face_mask_detection_model", save_format="h5")


#Use matplotlib to visual accuracy
#Graph loss/accuracy versus epoch #
plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.savefig(("plot.png"))



