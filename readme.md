This repository contains data files and scripts for developing an XLNet emotion recognition model with the python transformers library. Below is a summary of the files contained in the repository:

Training set.csv - Processed training data to give the model

BTD/CBET/SemEval Test.csv - Processed testing data, separated by the dataset they originally came from


combine_datasets.py - Processes the original datasets to produce a combined training set and separate test sets (the csv files in this directory)

train_model.py - A script that trains the model using examples from the Transformers library examples

helpers.py - Helper functions for combine_datasets.py and train_model.py


Emotion Recognition Datasets.zip - Original datasets for BTD, CBET, and SemEval 2018 Task 1
