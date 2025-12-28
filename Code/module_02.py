# Import necessary libraries and set environment variables to suppress warnings
import os

from torch import tensor
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf

# Creating tensors of different data types
print("\n=== Tensors with Different Data Types ===")
# String tensor
string = tf.Variable("this is a string", tf.string)
# Integer tensor
number = tf.Variable(324, tf.int16)
# Floating-point tensor
floating = tf.Variable(3.567, tf.float64)

# Print the tensors to see their representation
print(string)
print(number)
print(floating)

# rank Degree of tensor
print("\n=== Rank of Tensors ===")
rank1_tensor = tf.Variable(["Test"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
print("Rank 1 Tensor:", rank1_tensor)
print("Rank 2 Tensor:", rank2_tensor)

# Shape of tensors
print("\n=== Shape of Tensors ===")
rank3_tensor = tf.Variable(
    [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
    ],
    tf.int16,
)
print("Rank 3 Tensor Shape:", rank3_tensor.shape)

# change shape of tensor
print("\n=== Reshaping Tensors ===")
reshaped_tensor = tf.reshape(rank3_tensor, shape=(2, 2, 6))
print("Reshaped Tensor Shape:", reshaped_tensor.shape)
reshaped_tensor = tf.reshape(rank3_tensor, shape=(2, -1, 12))
print("Reshaped Tensor Shape:", reshaped_tensor.shape)
# populated with zeros
zero_tensor = tf.zeros([3, 4], tf.int16)
print("Zero Tensor:\n", zero_tensor)
# populated with ones
one_tensor = tf.ones([2, 3], tf.float64)
print("One Tensor:\n", one_tensor)
# populated with a specific value
filled_tensor = tf.fill([2, 4], 9)
print("Filled Tensor:\n", filled_tensor)
# random values between 0 and 1
random_tensor = tf.random.uniform([3, 4], minval=0, maxval=1)
print("Random Tensor:\n", random_tensor)
# random values from a normal distribution
normal_tensor = tf.random.normal([2, 3], mean=0.0, stddev=1.0)
print("Normal Tensor:\n", normal_tensor)    
# random values from a truncated normal distribution
truncated_tensor = tf.random.truncated_normal([2, 3], mean=0.0, stddev=1.0)
print("Truncated Normal Tensor:\n", truncated_tensor)
# Identity matrix
identity_matrix = tf.eye(4, 4)  
print("Identity Matrix:\n", identity_matrix)
# Sequence of numbers
sequence_tensor = tf.range(start=10, limit=30, delta=5)     
print("Sequence Tensor:\n", sequence_tensor)
# Linspace tensor
linspace_tensor = tf.linspace(start=0.0, stop=4.0, num=5)
print("Linspace Tensor:\n", linspace_tensor) 

# Casting tensors to different data types
print("\n=== Casting Tensors ===")
original_tensor = tf.constant([1.8, 2.2, 3.3], tf.float32)
int_tensor = tf.cast(original_tensor, tf.int32)
print("Original Tensor:", original_tensor)
print("Casted to Int Tensor:", int_tensor)  

# type of tensor
print("\n=== Data Types of Tensors ===")
# variable tensor
var_tensor = tf.Variable(["Test"], tf.string)
print("Variable Tensor Data Type:", var_tensor.dtype)
# constant tensor
const_tensor = tf.constant([1, 2, 3], tf.int16)
print("Constant Tensor Data Type:", const_tensor.dtype)
# placeholder tensor (using tf.function to simulate placeholder behavior)
@tf.function
def placeholder_tensor(x: tf.Tensor) -> tf.Tensor:
    return x
placeholder = placeholder_tensor(tf.constant([1.5, 2.5, 3.5], tf.float32))
print("Placeholder Tensor Data Type:", placeholder.dtype)
# sparse tensor
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
print("Sparse Tensor Data Type:", sparse_tensor.dtype)


# Evaluating tensors
print("\n=== Evaluating Tensors ===")
tensor = tf.constant([1, 2, 3])
print(tensor)        # prints the tensor
print(tensor.numpy())  # prints the actual values [1 2 3]