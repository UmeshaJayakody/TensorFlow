import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Define the initial distribution: probability of starting in each state
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])

# Transition distribution: probability of transitioning from one state to another
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])

# Observation distribution: normal distributions for each state
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# Create the Hidden Markov Model
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7
)

# Compute the mean of the observations
mean = model.mean()

# Print the mean
print(mean.numpy())