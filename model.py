# COMP-4740 Final Project - Computer Vision Model
# Created by Saffa Alvi and Nour ElKott

# The purpose of this research data analysis project is to apply deep learning approaches to explore Computer Vision
# annd create a model for the Dogs vs. Cats competition on Kaggle.com.

# The objective is to use the Kaggle provided dataset and write an algorithm to classify whether the images in the dataset are of a dog or a cat. 

# model.py - Creates the Convolutional Neural Network (CNN) model that is used to predict the dog and cat images. 
# The model is compiled with the Adam optimizer and it trains and tests based off dogs-vs-cats/train dataset. 
# Saves the label encodings (labels.npy), and the model in .json format (model.json) and as model.h5 to load the weights.


import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import sklearn
import cv2
from sklearn.preprocessing import LabelEncoder

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
np.save('labels.npy', lb.classes_)
print("Saved labels")

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
model.save('model.h5')
json_model = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json_model)

print("Saved model!")
