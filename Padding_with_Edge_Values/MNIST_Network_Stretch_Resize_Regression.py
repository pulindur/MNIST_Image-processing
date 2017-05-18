
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import cv2

from PIL import Image
from scipy.misc import imresize
from random import randint

batch_size = 10
num_classes = 10
epochs = 12

####################################################################

# Loading preprocessed Images

train_img = np.load('train_img_1_2.npy')
train_lab = np.load('train_lab.npy')

test_img = np.load('test_img_1_2.npy')
test_lab = np.load('test_lab.npy')

####################################################################

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    train_img = train_img.reshape(train_img.shape[0], 1, img_rows, img_cols)
    test_img  = test_img.reshape(test_img.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_img = train_img.reshape(train_img.shape[0], img_rows, img_cols, 1)
    test_img  = test_img.reshape(test_img.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

train_img = train_img.astype('float32')
test_img = test_img.astype('float32')
train_img /= 255
test_img /= 255
print('x_train shape:', train_img.shape)
print(train_img.shape[0], 'train samples')
print(test_img.shape[0], 'test samples')

# convert class vectors to binary class matrices
#train_lab = keras.utils.to_categorical(train_lab, num_classes)
#test_lab  = keras.utils.to_categorical(test_lab, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, input_dim=1, init='normal', activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_img, train_lab,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_img, test_lab))
score = model.evaluate(test_img, test_lab, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred_y = model.predict(test_img)

mse = mean_squared_error(pred_y, test_lab)
print('MSE:', mse)

pred_rnd = np.round(pred_y,0)
cnt = 0

for it in range(10000):
  if pred_rnd[it]==test_lab[it]:
    cnt = cnt+1 

acc      = cnt*0.0001

print('Prediction Accuracy:', acc)
print('Correct Count:', cnt)


np.save('pred_y_1_2.npy', pred_y)

