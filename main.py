import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.gridspec as gridspec
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout,Flatten, SimpleRNN,LSTM,SpatialDropout1D,RNN,LSTMCell,Activation,Conv1D,MaxPooling1D,GlobalMaxPooling1D,GRU
from sklearn.metrics import accuracy_score
from keras import optimizers
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import keras.backend as K

train_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1,validation_split=0.3)
path = 'data'
train_generator = train_datagen.flow_from_directory(
    path + '/train',
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    path + '/train',
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse',
    subset='validation')

def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')


def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)


K.clear_session()
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(28, 28, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.4))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=36, activation='softmax'))

# model = Sequential([
# Flatten(input_shape=(28, 28, 3)),
# Dense(128, activation='relu'),
# Dense(36, activation='softmax')
# ])
print (model.summary())

batch_size = 1
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001),  metrics=['accuracy'])
model_save_path='best_model.h5'
chekpoint_callback=ModelCheckpoint(model_save_path,monitor='val_accuracy',save_best_only=True,verbose=1)
result=model.fit(train_generator,epochs=20,steps_per_epoch=train_generator.samples // batch_size,validation_data=validation_generator,callbacks=[chekpoint_callback])

fig = plt.figure(figsize=(14,5))
grid=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
fig.add_subplot(grid[0])
plt.plot(result.history['accuracy'], label='Тренировочная точность')
plt.plot(result.history['val_accuracy'], label='Валидационная точность')
plt.title('Точность')
plt.xlabel('эпохи')
plt.ylabel('точность')
plt.legend()

fig.add_subplot(grid[1])
plt.plot(result.history['loss'], label='Тренировочные потери')
plt.plot(result.history['val_loss'], label='Валидационные потери')
plt.title('Потери')
plt.xlabel('эпохи')
plt.ylabel('потери')
plt.legend()
plt.show()

test_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.1, height_shift_range=0.1)
path = 'data'
test_generator = test_datagen.flow_from_directory(
    path + '/train',
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse')
print(model.evaluate(test_generator))

p=model.predict(test_generator)
p=np.argmax(p,axis=1)
print(test_generator.labels[:5],p[:5])
print(classification_report(test_generator.labels,p))