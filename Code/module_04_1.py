import os

# Suppress TensorFlow warnings and logs for cleaner output
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

# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist

# Split into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Display dataset information
print(f"Training images shape: {train_images.shape}")
print(f"Value of pixel at [0,23,23]: {train_images[0, 23, 23]}")
print(f"First 10 training labels: {train_labels[:10]}")

# Class names for the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Save a sample image
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.savefig('./graph_output/module_04/sample_image.png')
plt.close()

# Normalize the images to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
    keras.layers.Dense(128, activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)
print(f"Predictions array: {predictions[1]}")
print(f"Prediction for first test image: {np.argmax(predictions[1])}")
print(f"Actual label for first test image: {test_labels[1]}")

# plot the image and predictions
print(f'Predicted label: {class_names[np.argmax(predictions[1])]}')
plt.figure()
plt.imshow(test_images[1])
plt.colorbar()
plt.grid(False)
plt.savefig('./graph_output/module_04/sample_image_test.png')
plt.close()

# Verifying Predictions
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  print("Expected: " + label)
  print("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
