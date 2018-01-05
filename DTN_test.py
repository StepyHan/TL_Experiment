import time
import numpy as np
import tensorflow as tf

import DTN
# import DTN_train

import CWRU_Data

def evaluate(test_data, model_file):
    with tf.Graph().as_default() as g:
        with tf.device('cpu:0'):
            x = tf.placeholder(tf.float32, [
                test_data.num_examples,
                DTN.IMAGE_SIZE,
                DTN.IMAGE_SIZE,
                DTN.NUM_CHANNELS],
                               name='x-input')
            y_ = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='y-input')
            print(y_)

            _, y = DTN.inference(x, False, None, reuse=False, trainable=False)
            print(y)

            correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

            # load saved model

            saver = tf.train.Saver()

            with tf.Session() as sess:
                saver.restore(sess, model_file)
                accuracy_score = sess.run(accuracy, feed_dict={x: np.reshape(test_data.images,
                                                                             (test_data.num_examples,
                                                                              DTN.IMAGE_SIZE,
                                                                              DTN.IMAGE_SIZE,
                                                                              DTN.NUM_CHANNELS)
                                                                             ), y_: test_data.labels})
                print(model_file, "test accuracy = %f" % accuracy_score)


pre_model_file = 'model/PRE_DTN_CWRU_Data.ckpt'
dtn_model_file = 'model/DTN_CWRU_Data.ckpt'
x_3HP, y_3HP = CWRU_Data.get_data('CWRU/3HP')
target_data = CWRU_Data.Dataset(x_3HP, y_3HP)
source_data = CWRU_Data.Dataset(x_0HP, y_0HP)
# evaluate(target_data, pre_model_file)
evaluate(target_data, pre_model_file)

