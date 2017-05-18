
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

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

####################################################################

# Resizing Training images
trn_img = []
trn_hgt = 0 

for iter in range(0, 60000, 1):
    tmp    = x_train[iter]
       
    height = np.random.poisson(28*2,1) #* np.random.uniform(0.1,3)
    img    = imresize(tmp, (height, 28))
    trn_img.append(img)
    
    if trn_img[iter].size > trn_hgt:
        trn_hgt = trn_img[iter].size
        
###########################################################################

# Resizing Test images
tst_img = []
tst_hgt = 0 

for iter in range(0, 10000, 1):
    tmp    = x_test[iter]
        
    height = np.random.poisson(28*2,1) #* np.random.uniform(0.1,3)
    img    = imresize(tmp, (height, 28))
    tst_img.append(img)
    
    if tst_img[iter].size > tst_hgt:
        tst_hgt = tst_img[iter].size

max_size = max(trn_hgt, tst_hgt)


###########################################################################

# Creating Training Data Set
max_hgt = max_size/28

train_img = np.zeros((60000, max_hgt, 28), dtype='uint8')
train_lab = np.zeros((60000))

for i in range(60000):    #range(10):
  if max_hgt==trn_img[i].shape[0]:
      train_img[i,:,:] = trn_img[i]
      train_lab[i]     = y_train[i]
  else:
      top_rows = randint(1, (max_hgt - trn_img[i].shape[0]))
      bot_rows = max_hgt - (top_rows + trn_img[i].shape[0])
      
      train_img[i,:] = cv2.copyMakeBorder(trn_img[i],top_rows,bot_rows,0,0,cv2.BORDER_REPLICATE)
      train_lab[i]   = y_train[i]

###########################################################################

# Creating Test Data Set
test_img = np.zeros((10000, max_hgt, 28), dtype='uint8')
test_lab = np.zeros((10000))

for i in range(10000):    #range(10):
  if max_hgt==tst_img[i].shape[0]:
      test_img[i,:,:] = tst_img[i]
      test_lab[i]     = y_test[i]
  else:
      top_rows = randint(1, (max_hgt - tst_img[i].shape[0]))
      bot_rows = max_hgt - (top_rows + tst_img[i].shape[0])
      
      test_img[i,:] = cv2.copyMakeBorder(tst_img[i],top_rows,bot_rows,0,0,cv2.BORDER_REPLICATE)
      test_lab[i]   = y_test[i]

###########################################################################

# Saving Training and Test Data

np.save('train_img.npy', train_img)
np.save('train_lab.npy', train_lab)

np.save('test_img.npy', test_img)
np.save('test_lab.npy', test_lab)
