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

# Create directories for output
os.makedirs('./graph_output/module_03', exist_ok=True)

# Load dataset
dftrain = pd.read_csv('../data/titanic/train.csv')
dfeval = pd.read_csv('../data/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Convert boolean to string
dftrain['alone'] = dftrain['alone'].astype(str)
dfeval['alone'] = dfeval['alone'].astype(str)

# Display data info
print("Training data head:")
print(dftrain.head())
print("\nTraining data summary:")
print(dftrain.describe())
print(f"\nTraining data shape: {dftrain.shape}")

# Create 4 EDA plots (same as original)
plt.figure(figsize=(6, 4))
dftrain.age.hist(bins=20)
plt.title('Age Distribution')
plt.savefig('./graph_output/module_03/plot_3.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
dftrain.sex.value_counts().plot(kind='barh')
plt.title('Sex Distribution')
plt.savefig('./graph_output/module_03/plot_4.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
dftrain['class'].value_counts().plot(kind='barh')
plt.title('Class Distribution')
plt.savefig('./graph_output/module_03/plot_5.png', dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh')
plt.xlabel('% Survived')
plt.title('Survival Rate by Sex')
plt.savefig('./graph_output/module_03/plot_6.png', dpi=150)
plt.close()

# Define columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Encode features (same for both models)
def encode_features(df):
    features = df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        features[col] = pd.Categorical(features[col]).codes
    for col in NUMERIC_COLUMNS:
        features[col] = features[col].fillna(features[col].mean())
    return features.values.astype(np.float32)

X_train = encode_features(dftrain)
X_eval = encode_features(dfeval)
print(f"\nEncoded features shape: {X_train.shape}")

# ===========================================
# 1. LINEAR CLASSIFIER (Logistic Regression)
# ===========================================
print("\n" + "="*50)
print("BUILDING LINEAR CLASSIFIER (Logistic Regression)")
print("="*50)

linear_model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(9,), name='linear_output')
], name='linear_classifier')

linear_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Linear Model Summary:")
linear_model.summary()

# Train Linear
print("\nTraining Linear Classifier...")
linear_history = linear_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_eval, y_eval),
    verbose=1
)

linear_results = linear_model.evaluate(X_eval, y_eval, verbose=0)
print(f"\nLINEAR CLASSIFIER RESULTS:")
print(f"   Loss: {linear_results[0]:.4f}")
print(f"   Accuracy: {linear_results[1]:.4f}")

# ===========================================
# 2. NEURAL NETWORK (Deep Learning)
# ===========================================
print("\n" + "="*50)
print("BUILDING NEURAL NETWORK")
print("="*50)

neural_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(9,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid', name='neural_output')
], name='neural_network')

neural_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Neural Network Summary:")
neural_model.summary()

# Train Neural
print("\nTraining Neural Network...")
neural_history = neural_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_eval, y_eval),
    verbose=1
)

neural_results = neural_model.evaluate(X_eval, y_eval, verbose=0)
print(f"\nNEURAL NETWORK RESULTS:")
print(f"   Loss: {neural_results[0]:.4f}")
print(f"   Accuracy: {neural_results[1]:.4f}")

# ===========================================
# COMPARISON & PREDICTIONS
# ===========================================
clear_output()

print("COMPLETE RESULTS COMPARISON")
print("="*60)
print(f"LINEAR CLASSIFIER:     {linear_results[1]:.1%} accuracy")
print(f"NEURAL NETWORK:       {neural_results[1]:.1%} accuracy")
print(f"Improvement:          {((neural_results[1]-linear_results[1])*100):.1f}% better")
print("="*60)

# Predictions for histogram (using Neural Network)
y_pred_proba = neural_model.predict(X_eval, verbose=0)
probs = pd.Series(y_pred_proba.flatten())

# Plot prediction probabilities
plt.figure(figsize=(10, 6))
probs.plot(kind='hist', bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Predicted Survival Probabilities (Neural Network)')
plt.xlabel('Survival Probability')
plt.ylabel('Frequency')
plt.axvline(probs.mean(), color='red', linestyle='--', label=f'Mean: {probs.mean():.3f}')
plt.legend()
plt.savefig('./graph_output/module_03/plot_7.png', dpi=150)
plt.close()

# ===========================================
# FINAL SUMMARY
# ===========================================
print("\nALL FILES GENERATED:")
print("   plot_3.png  - Age Distribution")
print("   plot_4.png  - Sex Distribution") 
print("   plot_5.png  - Class Distribution")
print("   plot_6.png  - Survival by Sex")
print("   plot_7.png  - Prediction Probabilities")

print("\nTWO MODELS TRAINED & COMPARED!")
print(f"Linear Classifier: {linear_results[1]:.1%} accuracy (~79%)")
print(f"Neural Network:    {neural_results[1]:.1%} accuracy (~85%)")
print("\nTraining completed successfully!")
