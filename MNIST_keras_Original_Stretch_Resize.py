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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import cv2

from PIL import Image
from scipy.misc import imresize
from random import randint
from tempfile import TemporaryFile

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

############################################################
# Stretching images to 89x28

trn_img = np.zeros((60000, 89, 28), dtype='uint8')

for it in range(60000):
    tmp               = x_train[it]
    img               = imresize(tmp, (89, 28))
    trn_img[it,:,:] = img

tst_img = np.zeros((10000, 89, 28), dtype='uint8')

for it in range(10000):
    tmp             = x_test[it]
    img             = imresize(tmp, (89, 28))
    tst_img[it,:,:] = img
    
############################################################
# Resizing stretched images to 28x28

x_train1 = np.zeros((60000, 28, 28), dtype='uint8')

for it in range(60000):
    tmp              = trn_img[it,:,:]
    img              = imresize(tmp, (28, 28))
    x_train1[it,:,:] = img

x_test1 = np.zeros((10000, 28, 28), dtype='uint8')

for it in range(10000):
    tmp             = tst_img[it,:,:]
    img             = imresize(tmp, (28, 28))
    x_test1[it,:,:] = img

############################################################

if K.image_data_format() == 'channels_first':
    x_train1 = x_train1.reshape(x_train1.shape[0], 1, img_rows, img_cols)
    x_test1 = x_test1.reshape(x_test1.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train1 = x_train1.reshape(x_train1.shape[0], img_rows, img_cols, 1)
    x_test1 = x_test1.reshape(x_test1.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train1 = x_train1.astype('float32')
x_test1 = x_test1.astype('float32')
x_train1 /= 255
x_test1 /= 255
print('x_train shape:', x_train1.shape)
print(x_train1.shape[0], 'train samples')
print(x_test1.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train1, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test1, y_test))
score = model.evaluate(x_test1, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

