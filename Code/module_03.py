from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import necessary libraries for plotting and numerical operations
import matplotlib.pyplot as plt
import numpy as np

# Define sample data points for x and y coordinates
x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]

# Create and save a basic scatter plot
plt.figure()
plt.plot(x, y, 'ro')  # Plot red circles at each (x,y) point
plt.axis([0, 6, 0, 20])  # Set x-axis from 0-6, y-axis from 0-20
plt.savefig('./graph_output/module_03/plot_1.png')  # Save the scatter plot as PNG
plt.close()

# Create and save a scatter plot with linear regression line
plt.figure()
plt.plot(x, y, 'ro')  # Plot the same red circles
plt.axis([0, 6, 0, 20])  # Set axis limits again
# Fit a linear polynomial to the data and plot the regression line
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.savefig('./graph_output/module_03/plot_2.png')  # Save the plot with line as PNG
plt.close()

# Import necessary libraries for data handling and TensorFlow
import pandas as pd
from IPython.display import clear_output
import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('../data/titanic/train.csv') # training data
dfeval = pd.read_csv('../data/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# display first few rows of training data
print(dftrain.head())
# get summary statistics of training data
print(dftrain.describe())
# get the shape of training data
print(dftrain.shape)

plt.figure()
dftrain.age.hist(bins=20)  # Plot histogram of 'age' feature
plt.savefig('./graph_output/module_03/plot_3.png')  # Save histogram as
plt.close()