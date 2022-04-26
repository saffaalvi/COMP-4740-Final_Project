# COMP-4740 Final Project - Computer Vision Model
# Created by Saffa Alvi and Nour ElKott

# The purpose of this research data analysis project is to apply deep learning approaches to explore Computer Vision
# annd create a model for the Dogs vs. Cats competition on Kaggle.com.

# The objective is to use the Kaggle provided dataset and write an algorithm to classify whether the images in the dataset are of a dog or a cat. 

# test.py - Makes predictions on the dogs-vs-cats/test1 dataset from the model previously created in model.py
# Also provides an application of our model to the dataset where user can see the image and the model prediction.


import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import cv2
import random
import matplotlib.pyplot as plt

# Load the previously saved model
json_model_file = open('model.json', 'r')
json_model = json_model_file.read()
json_model_file.close()
model = model_from_json(json_model)
model.load_weights('model.h5')
print("Loaded model!")

# Load the labels
labels = np.load('labels.npy', allow_pickle=True)
labels = list(labels)
#labels = ["cat", "dog"]
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
print("Test data shape:", test_data.shape)

predictions = model.predict(test_data)
print("The model has predicted the test data. This is a sample before rounding:", predictions[:3])

# to get as classes (since we used sigmoid, will have to round)
classes = []
for p in predictions:
  classes.append(round(p[0]))

print("After rounding:", classes[:3])

# Show image and model prediction

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
