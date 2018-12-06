import tensorflow as tf

session = tf.Session()

hello = tf.constant("Hello TensorFlow")
print(session.run(hello))

a = tf.constant(2)
b = tf.constant(2)
c = tf.constant(1)

print('{0} plus {1} is {2} minus {3} that\'s {4}, quick maths'.format(*[session.run(x) for x in [a, b, a+b, c, a+b-c]]))