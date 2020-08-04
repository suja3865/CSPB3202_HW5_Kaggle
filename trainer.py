import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2

''' File structure:
data:
    - train:
            - .tif files
    -test:
          - tif files
    -train_labels.csv
    
    -sample_submission.csv
'''

# Options for image size, training and testing sizes

img_height = 96
img_width = 96

data_size = 15000
validation_split = 0.3

train_size = int((1-validation_split)*data_size)
test_size = int(validation_split*data_size)

train_data = []
train_labels = []

test_data = []
test_labels = []

# Store data as pandas data frame

data = pd.read_csv('data/train_labels.csv')

# Read images and labels into an array

for i in range(data_size):
    img_array = cv2.imread(os.path.join("data/train",data['id'][i]+'.tif'),cv2.IMREAD_GRAYSCALE)
    train_data.append(img_array)
    train_labels.append(data['label'][i])

# Split training and testing data according to validation split
    
test_data = train_data[train_size:]
test_labels = train_labels[train_size:]

train_data = train_data[:train_size]
train_labels = train_labels[:train_size]

# Process training and testing data into numpy by flattening the arrays and normalizing values between 0 to 1

train_data = np.array(train_data)
test_data = np.array(test_data)

train_data = np.reshape(train_data, (train_size, img_height*img_width))
test_data = np.reshape(test_data, (test_size, img_height*img_width))

train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

train_data /= 255
test_data /= 255

# Perform hot encoding on categories

train_labels_  = tf.keras.utils.to_categorical(train_labels, 2)
test_labels_  = tf.keras.utils.to_categorical(test_labels, 2)

# Build and compile a model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_dim=img_height*img_width))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

# We will now compile and print out a summary of our model
opt = tf.keras.optimizers.SGD(learning_rate=0.5)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

# Train for selected number of epochs
model.fit(train_data, train_labels_, epochs=10)

# Test model against testing data
loss, accuracy = model.evaluate(test_data, test_labels_)
print('Test accuracy: %.2f' % (accuracy))

model.save('model1.model')



