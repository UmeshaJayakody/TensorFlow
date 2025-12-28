from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython.display import clear_output

# Define constants
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Download datasets
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# Load data
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

print("Train shape:", train.shape)
print(train.head())

# FIXED: Encode labels (Species to 0,1,2) - No .values on numpy array
def encode_labels(y):
    return pd.Categorical(y).codes.astype(np.float32)

train_y_encoded = encode_labels(train_y)
test_y_encoded = encode_labels(test_y)

# Prepare features (all numeric, no preprocessing needed)
X_train = train.values.astype(np.float32)
X_test = test.values.astype(np.float32)

print("\nEncoded features shape:", X_train.shape)

# ===========================================
# 1. LINEAR CLASSIFIER (Logistic Regression)
# ===========================================
print("\n" + "="*50)
print("BUILDING LINEAR CLASSIFIER")
print("="*50)

linear_model = keras.Sequential([
    layers.Dense(3, activation='softmax', input_shape=(4,), name='linear_output')
], name='iris_linear')

linear_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Linear Model Summary:")
linear_model.summary()

# Train Linear
print("\nTraining Linear Classifier...")
linear_model.fit(
    X_train, train_y_encoded,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, test_y_encoded),
    verbose=1
)

linear_results = linear_model.evaluate(X_test, test_y_encoded, verbose=0)
print(f"\nLINEAR CLASSIFIER RESULTS:")
print(f"   Loss: {linear_results[0]:.4f}")
print(f"   Accuracy: {linear_results[1]:.4f}")

# ===========================================
# 2. DNN (Deep Neural Network) - Replaces tf.estimator.DNNClassifier
# ===========================================
print("\n" + "="*50)
print("BUILDING DEEP NEURAL NETWORK (DNN)")
print("="*50)

dnn_model = keras.Sequential([
    layers.Dense(30, activation='relu', input_shape=(4,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax', name='dnn_output')
], name='iris_dnn')

dnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("DNN Model Summary:")
dnn_model.summary()

# Train DNN
print("\nTraining DNN...")
dnn_model.fit(
    X_train, train_y_encoded,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, test_y_encoded),
    verbose=1
)

dnn_results = dnn_model.evaluate(X_test, test_y_encoded, verbose=0)
print(f"\nDNN RESULTS:")
print(f"   Loss: {dnn_results[0]:.4f}")
print(f"   Accuracy: {dnn_results[1]:.4f}")

# ===========================================
# PREDICTIONS (Interactive + Example)
# ===========================================
print("\n" + "="*50)
print("PREDICTIONS")
print("="*50)

# Example predictions
predict_x = {
    'SepalLength': np.array([5.1, 5.9, 6.9]).reshape(-1, 1),
    'SepalWidth': np.array([3.3, 3.0, 3.1]).reshape(-1, 1),
    'PetalLength': np.array([1.7, 4.2, 5.4]).reshape(-1, 1),
    'PetalWidth': np.array([0.5, 1.5, 2.1]).reshape(-1, 1)
}

X_example = np.hstack([predict_x['SepalLength'], predict_x['SepalWidth'], 
                      predict_x['PetalLength'], predict_x['PetalWidth']])

print("Example predictions (DNN):")
example_pred = dnn_model.predict(X_example)
for i, pred in enumerate(example_pred):
    class_id = np.argmax(pred)
    probability = pred[class_id]
    print(f"Input {i+1}: Prediction is \"{SPECIES[class_id]}\" ({probability:.1%})")

# Interactive prediction
print("\n" + "-"*30)
print("INTERACTIVE PREDICTION (type 4 numbers):")
print("Please type numeric values as prompted.")

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = []

for feature in features:
    while True:
        try:
            val = float(input(f"{feature}: "))
            predict.append(val)
            break
        except ValueError:
            print("Please enter a valid number.")

predict_array = np.array(predict).reshape(1, -1)
pred_result = dnn_model.predict(predict_array, verbose=0)
class_id = np.argmax(pred_result[0])
probability = pred_result[0][class_id]

print(f"\nPrediction is \"{SPECIES[class_id]}\" ({probability:.1%})")

# ===========================================
# FINAL COMPARISON
# ===========================================
print("\n" + "="*60)
print("FINAL RESULTS COMPARISON")
print("="*60)
print(f"LINEAR CLASSIFIER: {linear_results[1]:.1%} accuracy")
print(f"DNN CLASSIFIER:    {dnn_results[1]:.1%} accuracy")
print(f"Improvement:       {((dnn_results[1]-linear_results[1])*100):.1f}% better")
print("="*60)
