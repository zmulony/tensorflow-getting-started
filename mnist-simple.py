import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### DEFINE MODEL

# 784 = 28 x 28 which are the dimensions of the MNIST data
x = tf.placeholder(tf.float32, shape=[None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.placeholder(tf.float32, shape=[None, 10])

y = tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

### RUN

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

training_steps = 1000

for epoch in range(training_steps):
  batch_size = 100

  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={ x: batch_xs, y_: batch_ys })

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels })
print("Test accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()