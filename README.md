# Computer Vision (CNN)
**COMP-4740 Final Project** 

_Submitted by: Saffa Alvi and Nour ElKott_ 

This repository contains the source code for a Convolutional Neural Network (CNN) Model for the Dogs vs. Cats competition on Kaggle.com. In order to test this code, Tensorflow needs to be installed. A list of required packages is also given in requirements.txt

We were unable to provide the dataset in this repositroy as it was too large. If trying to test this locally, will have to download the dogs-vs-cats dataset from Kaggle found here: 
[https://www.kaggle.com/competitions/dogs-vs-cats/data](https://www.kaggle.com/competitions/dogs-vs-cats/data)

Extract train.zip and test1.zip into a dogs-vs-cats directory.

- **final_project.py** - the source code for our CNN model implementation. 
- **dogs-vs-cats.ipynb** - a Jupyter notebook file that contains the outputs and some notes for our model
- **model.py**- creates the Convolutional Neural Network (CNN) model that is used to predict the dog and cat images. 
- **test.py** - makes predictions on the dogs-vs-cats/test1 dataset from the model previously created in model.py
- **model.h5 and model.json** - saved CNN model without weights (easier and quicker to load and use when testing)
- **labels.npy** - encoded classes/labels from dataset (cats and dogs) 
- **requirements.txt, Pipfile, Piplock**: All the required packages and versions needed to run the program files.
