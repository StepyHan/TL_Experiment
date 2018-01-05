import os
import numpy as np

import tensorflow as tf
import DTN
import scipy.io as scio
from sklearn.cross_validation import train_test_split

import CWRU_Data

PRE_BATCH_SIZE = 100
BATCH_SIZE = 300
RAGULARZTION_RATE = 0.0001
LEARNING_RATE = 0.01

PRE_TRAINING_STEPS = 3000
TRAINING_STEPS = 20

theta1 = 10
theta2 = 10


MODEL_SAVE_PATH = "model/"
PRE_MODEL_NAME = "PRE_DTN_CWRU_Data.ckpt"
MODEL_NAME = "DTN_CWRU_Data.ckpt"

# DTN.OUTPUT_NODE number of class

def pre_train(source_domain_data):
    xs = tf.placeholder(tf.float32, [PRE_BATCH_SIZE,
                                    DTN.IMAGE_SIZE,
                                    DTN.IMAGE_SIZE,
                                    DTN.NUM_CHANNELS],
                       name='source-x-input')

    ys_ = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='source-y-input')
    regularizer = tf.contrib.layers.l2_regularizer(RAGULARZTION_RATE)
    _, ys = DTN.inference(xs, True, regularizer, reuse=False, trainable=True)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys, labels=tf.argmax(ys_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(PRE_TRAINING_STEPS):
            x, y = source_domain_data.next_batch(PRE_BATCH_SIZE)
            reshaped_xs = np.reshape(x, (PRE_BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))
            loss_value, _ = sess.run([loss, train], feed_dict={xs: reshaped_xs, ys_: y})
            print("After %d training steps, loss on training batch is %f" % (i, loss_value))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, PRE_MODEL_NAME))


# def adapatation_train(source_domain_data, target_domain_data):
#     with tf.Graph().as_default() as g:
#         with tf.device('cpu:0'):
#             xs = tf.placeholder(tf.float32, [BATCH_SIZE,
#                                             DTN.IMAGE_SIZE,
#                                             DTN.IMAGE_SIZE,
#                                             DTN.NUM_CHANNELS],
#                                name='source-x-input')
#
#             xt = tf.placeholder(tf.float32, [BATCH_SIZE,
#                                              DTN.IMAGE_SIZE,
#                                              DTN.IMAGE_SIZE,
#                                              DTN.NUM_CHANNELS],
#                                 name='target-x-input')
#
#             ys_ = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='source-y-input')
#
#             regularizer = None
#             source_fc, ys = DTN.inference(xs, False, regularizer, reuse=False)      # False: without dropout
#             target_fc, yt = DTN.inference(xt, False, regularizer, reuse=True)      # pesudo logit: yt
#
#             cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys, labels=tf.argmax(ys_, 1))
#             # 加上target y中有标签的部分
#             cross_entropy_mean = tf.reduce_mean(cross_entropy)
#
#             mmd_mar = tf.reduce_sum(
#                 (tf.reduce_mean(source_fc, 0) - tf.reduce_mean(target_fc, 0))
#                 * (tf.reduce_mean(source_fc, 0) - tf.reduce_mean(target_fc, 0)))
#
#             mmd_con = tf.reduce_sum(
#                 (tf.reduce_mean(ys, 0) - tf.reduce_mean(yt, 0))
#                 * (tf.reduce_mean(ys, 0) - tf.reduce_mean(yt, 0)))
#
#             loss = cross_entropy_mean + theta1 * mmd_mar + theta2 * mmd_con
#
#             train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
#
#
#             with tf.Session() as sess:
#                 saver = tf.train.Saver()
#                 saver.restore(sess, os.getcwd() + '/model/PRE_DTN_CWRU_Data.ckpt')
#                 for i in range(TRAINING_STEPS):
#                     print(i)
#                     source_X, source_Y = source_domain_data.next_batch(BATCH_SIZE)
#                     target_X, target_Y = target_domain_data.next_batch(BATCH_SIZE)
#
#                     reshaped_xs = np.reshape(source_X, (BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))
#                     reshaped_xt = np.reshape(target_X, (BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))
#
#                     yys, yyt, loss_value, _, MMD_MAR, MMD_CON = sess.run([ys, yt, loss, train, mmd_mar, mmd_con], feed_dict={xs: reshaped_xs,
#                                                                                                            ys_: source_Y,
#                                                                                                            xt: reshaped_xt}
#                                                                )
#                     print("yys:", yys, "yyt:", yyt)
#                     print("After %d training steps, loss on training batch is %f, mmd_mar is %f, mmd_con is %f" %
#                           (i, loss_value, MMD_MAR, MMD_CON))
#                 new_saver = tf.train.Saver()
#                 new_saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))



x_0HP, y_0HP = CWRU_Data.get_data('CWRU/0HP')
x_3HP, y_3HP = CWRU_Data.get_data('CWRU/3HP')
train_X_3HP, test_X_3HP, train_y_3HP, test_y_3HP = train_test_split(x_3HP, y_3HP, test_size=0.3, random_state=0)
source_domain_data = CWRU_Data.Dataset(x_0HP, y_0HP)
target_domain_data = CWRU_Data.Dataset(train_X_3HP, train_y_3HP)

pre_train(source_domain_data)
# adapatation_train(source_domain_data, target_domain_data)     # 第二次训练，reuse

