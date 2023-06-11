# importing the required libraries
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.datasets import mnist
# Set TensorFlow log level to suppress unnecessary messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib.pyplot as plt
#%matplotlib inline

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print some information about the dataset
print('Training data shape:', x_train.shape)
print('Number of training labels:', len(y_train))
print('Testing data shape:', x_test.shape)
print('Number of testing labels:', len(y_test))
print('Number of test image:', len(x_test))

# Display sample images and their labels
fig, axes = plt.subplots(5, 6, figsize=(8, 6))
axes = axes.flatten()
for i in range(30):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].set_title(f"label: {y_train[i]}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# number of neurons in input layer
num_input = len(x_test)

# number of neurons in hidden layer 1
num_hidden_1 = len(x_test)

# number of neurons in hidden layer 2
num_hidden_2 = int(num_hidden_1 / 2)

# number of neurons in hidden layer 3
num_hidden_3 = int(num_hidden_2 / 2)

# number of neurons in output layer
num_output = 10

print('Number of input    = {}'.format(num_input))

print('Number of hidden 1 = {}'.format(num_hidden_1))

print('Number of hidden 2 = {}'.format(num_hidden_2))

print('Number of hidden 3 = {}'.format(num_hidden_3))

print('Number of output   = {}'.format(num_output))

# placeholder for input
with tf.name_scope('input'):
    X = tf.placeholder("float", [None, num_input])

# placeholder for output
with tf.name_scope('output'):
    Y = tf.placeholder("float", [None, num_output])

# placeholders and initializations of the weigths
with tf.name_scope('weights'):
    weights = {
        'w1': tf.Variable(tf.truncated_normal([num_input, num_hidden_1], stddev = 0.1), name = 'weight_1'),
        'w2': tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2], stddev = 0.1), name = 'weight_2'),
        'w3': tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_3], stddev = 0.1), name = 'weight_3'),
        'w4': tf.Variable(tf.truncated_normal([num_hidden_3, num_output], stddev = 0.1), name = 'weight_4'),
    }

# placeholders and initializations of the bias
with tf.name_scope('biases'):
    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape = [num_hidden_1]), name = 'bias_1'),
        'b2': tf.Variable(tf.constant(0.1, shape = [num_hidden_2]), name = 'bias_2'),
        'b3': tf.Variable(tf.constant(0.1, shape = [num_hidden_3]), name = 'bias_3'),
        'b4': tf.Variable(tf.constant(0.1, shape = [num_output]), name = 'bias_4'),
    }