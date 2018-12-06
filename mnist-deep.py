import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### DEFINE MODEL
# 784 = 28 x 28 which are the dimensions of the MNIST data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, shape=[-1, 28, 28, 1], name="x_image")

### RUN
sess = tf.InteractiveSession()