import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
#import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sklearn.metrics import f1_score
from keras.models import load_model
from keras import optimizers
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D

# Create a new model instance
loaded_model = Sequential()
loaded_model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 3), activation='relu'))
loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
loaded_model.add(Dropout(rate=0.4))
loaded_model.add(Flatten())
loaded_model.add(Dense(units=128, activation='relu'))
loaded_model.add(Dense(units=36, activation='softmax'))

loaded_model.load_weights('best_model.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data'
test_generator = test_datagen.flow_from_directory(
    path + '/train',
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse',
    subset='training')
t=loaded_model.evaluate_generator(test_generator)
print(t)

p=loaded_model.predict_generator(test_generator)
p=np.argmax(p,axis=1)
print(f1_score(test_generator.labels,p))

