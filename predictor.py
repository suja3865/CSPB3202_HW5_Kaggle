import os
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

imgs = os.listdir("data/test")

submission = pd.DataFrame({'id':[],'label':[]})

model = tf.keras.models.load_model("model1.model")

labels = [0, 1]

for img in imgs:
    img_array = cv2.imread(os.path.join("data/test", img), cv2.IMREAD_GRAYSCALE)
    img_array = np.array(img_array)
    img_array=np.reshape(img_array, (1, 96*96))
    img_array = img_array.astype(np.float32)

    img_array /= 255

    predict = model.predict(img_array)
    if (predict[0][0] >= predict[0][1]):
        prediction = 0
    else:
        prediction = 1

    submission.loc[len(submission)] = [img[:len(img)-4], prediction]

submission.to_csv('submission.csv')
