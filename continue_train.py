import os
import numpy as np

import tensorflow as tf
import DTN
import scipy.io as scio
from sklearn.cross_validation import train_test_split

import CWRU_Data

BATCH_SIZE = 2000
RAGULARZTION_RATE = 0.0001
LEARNING_RATE = 0.01

TRAINING_STEPS = 100

theta1 = 0.05
theta2 = 0.05

# TODO continue train by fix lower layer and update adapatation layer !!!

MODEL_SAVE_PATH = "model/"
PRE_MODEL_NAME = "PRE_DTN_CWRU_Data.ckpt"
MODEL_NAME = "DTN_CWRU_Data.ckpt"

x_0HP, y_0HP = CWRU_Data.get_data('CWRU/0HP')
x_3HP, y_3HP = CWRU_Data.get_data('CWRU/3HP')
# train_X_3HP, test_X_3HP, train_y_3HP, test_y_3HP = train_test_split(x_3HP, y_3HP, test_size=0.3, random_state=0)
source_domain_data = CWRU_Data.Dataset(x_0HP, y_0HP)
target_domain_data = CWRU_Data.Dataset(x_3HP, y_3HP)

with tf.Graph().as_default() as g:
    with tf.device('cpu:0'):

        xs = tf.placeholder(tf.float32, [BATCH_SIZE,
                                         DTN.IMAGE_SIZE,
                                         DTN.IMAGE_SIZE,
                                         DTN.NUM_CHANNELS],
                            name='source-x-input')

        xt = tf.placeholder(tf.float32, [BATCH_SIZE,
                                         DTN.IMAGE_SIZE,
                                         DTN.IMAGE_SIZE,
                                         DTN.NUM_CHANNELS],
                            name='target-x-input')

        ys_ = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='source-y-input')

        # trainable=False fix lower layer param
        source_fc, ys = DTN.inference(xs, False, None, reuse=False, trainable=False)  # False: without dropout

        target_fc, yt = DTN.inference(xt, False, None, reuse=True, trainable=False)  # pesudo logit: yt

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys, labels=tf.argmax(ys_, 1))
        # 加上target y中有标签的部分
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        mmd_mar = tf.reduce_sum(tf.multiply((tf.reduce_mean(source_fc, 0) - tf.reduce_mean(target_fc, 0)),
                                            (tf.reduce_mean(source_fc, 0) - tf.reduce_mean(target_fc, 0))))

        mmd_con = tf.reduce_sum(tf.multiply((tf.reduce_mean(ys, 0) - tf.reduce_mean(yt, 0)),
                                            (tf.reduce_mean(ys, 0) - tf.reduce_mean(yt, 0))))

        loss = cross_entropy_mean + theta1 * mmd_mar + theta2 * mmd_con

        train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, os.getcwd() + '/model/PRE_DTN_CWRU_Data.ckpt')
            for i in range(TRAINING_STEPS):
                source_X, source_Y = source_domain_data.next_batch(BATCH_SIZE)
                target_X, target_Y = target_domain_data.next_batch(BATCH_SIZE)

                reshaped_xs = np.reshape(source_X, (BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))
                reshaped_xt = np.reshape(target_X, (BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))

                cross_entropy_loss, yys, yyt, loss_value, _, MMD_MAR, MMD_CON = sess.run\
                    ([cross_entropy_mean, ys, yt, loss, train, mmd_mar, mmd_con],
                     feed_dict={xs: reshaped_xs,
                                ys_: source_Y,
                                xt: reshaped_xt}
                     )
                # print("yys:", yys)
                # print("yyt:", yyt)
                print("After %d training steps, loss on training batch is %f, "
                      "mmd_mar is %f, mmd_con is %f, cross_entropy os %f" %
                      (i, loss_value, MMD_MAR, MMD_CON, cross_entropy_loss))
            new_saver = tf.train.Saver()
            new_saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))


