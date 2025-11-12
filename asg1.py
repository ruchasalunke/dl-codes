import tensorflow as tf

# Print TensorFlow version
print(tf.__version__)

# Test TensorFlow computation
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
from tensorflow import keras
from keras.datasets import mnist

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check dataset shape
print("Training images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

!pip install pymc theano-pymc

!pip install aesara
import aesara
aesara.config.mode = "FAST_COMPILE"
aesara.config.cxx = ""  

import aesara.tensor as T
from aesara import function

# Declare symbolic variables
x = T.dscalar('x')
y = T.dscalar('y')

# Define an expression
z = x + y

# Compile the function
f = function([x, y], z)
# Test program for PyTorch

# The usual imports
import torch
import torch.nn as nn

# Print out the PyTorch version used
print("PyTorch version:", torch.__version__)

# Simple test to check if PyTorch is working properly
x = torch.rand(3, 3)
print("\nRandom tensor:\n", x)

# Check if CUDA (GPU) is available
print("\nCUDA available:", torch.cuda.is_available())

!pip install torch

!pip install tensorflow

# Test
print(f(5, 7))
