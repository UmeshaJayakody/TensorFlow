import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training sets

train_images.shape
print(train_images.shape)
print(train_images[0,23,23])  # let's have a look at one pixel
print(train_labels[:10])  # first 10 labels

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False) 
plt.savefig('./graph_output/module_04/sample_image.png')
plt.close()