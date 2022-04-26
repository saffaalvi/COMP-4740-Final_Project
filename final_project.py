# COMP-4740 Final Project - Computer Vision Model
# Created by Saffa Alvi and Nour ElKott

# The purpose of this research data analysis project is to apply deep learning approaches to explore Computer Vision
# annd create a model for the Dogs vs. Cats competition on Kaggle.com.

# The objective is to use the Kaggle provided dataset and write an algorithm to classify whether the images in the dataset are of a dog or a cat. 

# final_project.py - contains all the source code for this project. We have split it up into model.py and test.py since the model takes 
# several minutes (based on machine) to run. To avoid having to run it each time, we ran it once and saved the model so it can be loaded to make predictions.

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import os
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import random

# Dataset to read from
dataset_path = "dogs-vs-cats"
train_dataset = "dogs-vs-cats/train"
files = os.listdir(train_dataset)

# Make sure all files properly loaded
print(len(files))

# Training Dataset
train_x = []
train_y = []

# Iterate through files in training dataset
for f in files:
  # Get label/animal from filename - ex. cat.0.jpg, label is before .
  label = f.split(".")[0]
  filepath = train_dataset + "/" + f                  # image filepath
  data = imageio.imread(filepath, as_gray = True)     # read the specified image (in greyscale)
  data_arr = cv2.resize(data, dsize = (80, 80))       # resize the image to 90x90
  # add image data and label to respective arrays
  train_x.append(data_arr)
  train_y.append(label)

# Training Data
print(len(train_x))
print(len(train_y))

# Normalize/Scale Data and Change Dimensions
X_train = tf.keras.utils.normalize(train_x, axis=1) 

# need to reshape the data as keras needs 4D datasets, and ours are 3D right now
X_train = np.expand_dims(X_train, axis=-1)

# new reshaped dataset
print(X_train.shape)

# Label Encoding (Emotions -> Numbers)
# Need to change the string values of animals (cat or dog) into numbers so the CNN can properly predict them based on classes/labels.

lb = LabelEncoder()
Y_train = lb.fit_transform(train_y)
labels = lb.classes_
np.save('labels2.npy', lb.classes_)

# ----------------------- CONVOLUTIONAL NEURAL NETWORK MODEL -----------------------

model = keras.Sequential()

# add the layers
# hidden layers
model.add(keras.layers.Conv2D(128, 3, input_shape=(80, 80, 1), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
tf.keras.layers.Dropout(0.30)

model.add(keras.layers.Conv2D(256, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
tf.keras.layers.Dropout(0.40)

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# ----------------------- CONVOLUTIONAL NEURAL NETWORK MODEL -----------------------

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32)

# Save the model
model.save('model2.h5')
json_model = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(json_model)

print("Saved model!")

# Load the previously saved model
json_model_file = open('model2.json', 'r')
json_model = json_model_file.read()
json_model_file.close()
model = model_from_json(json_model)
model.load_weights('model2.h5')
print("Loaded model!")

# Load the labels
labels = np.load('labels2.npy', allow_pickle=True)
labels = list(labels)
print("Loaded labels:", labels)

# Test Data Preparation
test_data = []
test_data_image = []
test_dataset = "dogs-vs-cats/test1"
files = os.listdir(test_dataset)

for f in files:
  id = f.split(".")[0]
  filepath = test_dataset + "/" + f
  data = imageio.imread(filepath, as_gray = True)     # read the specified image (in greyscale)
  data_arr = cv2.resize(data, dsize = (80, 80))       # resize the image to 90x90
  test_data_image.append(data_arr)

test_data = tf.keras.utils.normalize(test_data_image, axis=1) 

# need to reshape the data as keras needs 4D datasets, and ours are 3D right now
test_data = np.expand_dims(test_data, axis=-1)
# new reshaped dataset
print(test_data.shape)

predictions = model.predict(test_data)
print(predictions)

# to get as classes (since we used sigmoid, will have to round)
classes = []
for p in predictions:
  classes.append(round(p[0]))

size = len(test_data_image)

while(1):
    image = random.randint(0, size)
    plt.imshow(test_data_image[image], cmap="gray")
    plt.show()

    model_prediction = int(classes[image])

    print("Model prediction - class:", model_prediction, "which is a", labels[model_prediction])

    # Pause when 'q' is entered
    cont = input('Paused - press ENTER to continue, q to exit: ')
    if cont == 'q':
        break
