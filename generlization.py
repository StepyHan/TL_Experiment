import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# TODO a new regularzation function is dedfined


plt.plot(np.linspace(-10, 10), np.exp(-np.square(np.linspace(-10, 10))))

def new_regularizer(weights):

    weights = tf.constant([[1,2,3], [4,5,6]], dtype=tf.float32)
    b = tf.exp(-tf.square(weights))
    a = 10 * tf.reduce_sum(b)

    with tf.Session() as sess:
        print(a.eval())