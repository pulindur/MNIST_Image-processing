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

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

###############################################################

x_train_bin = np.zeros((60000, 28, 28, 2), dtype='uint8')
x_test_bin  = np.zeros((10000, 28, 28, 2), dtype='uint8')

x_train_bin[:,:,:,0] = x_train
x_test_bin[:,:,:,0]  = x_test

for it in range(60000):
  temp_tr = x_train[it,:,:]>0
  x_train_bin[it,:,:,1] = temp_tr.astype(int)

for it in range(10000):
  temp_ts = x_test[it,:,:]>0
  x_test_bin[it,:,:,1] = temp_ts.astype(int)

###############################################################

if K.image_data_format() == 'channels_first':
    x_train_bin = x_train_bin.reshape(x_train_bin.shape[0], 2, img_rows, img_cols)
    x_test_bin = x_test_bin.reshape(x_test_bin.shape[0], 2, img_rows, img_cols)
    input_shape = (2, img_rows, img_cols)
else:
    x_train_bin = x_train_bin.reshape(x_train_bin.shape[0], img_rows, img_cols, 2)
    x_test_bin = x_test_bin.reshape(x_test_bin.shape[0], img_rows, img_cols, 2)
    input_shape = (img_rows, img_cols, 2)

x_train_bin = x_train_bin.astype('float32')
x_test_bin = x_test_bin.astype('float32')
x_train_bin /= 255
x_test_bin /= 255
print('x_train shape:', x_train_bin.shape)
print(x_train_bin.shape[0], 'train samples')
print(x_test_bin.shape[0], 'test samples')

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

model.fit(x_train_bin, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_bin, y_test))
score = model.evaluate(x_test_bin, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

