# CSPB3202_HW5_Kaggle
Repository for cspb 3202 hw5 kaggle:

Download data from kaggle and rename the folder containg all the data as "data"

trainer.py: Loads data, performs preprocessing on the images and trains the model. Change variables to change number of images used, valdiation split for testing.

model1.model: Sequential model model trained by trainer.py

predictor.py: Using model1.model to predict the the labels for images in the test folder and generate a submission.csv with the results.

Run using python3 trainer.py or python3 predictor.py
