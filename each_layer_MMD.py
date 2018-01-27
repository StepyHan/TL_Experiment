import os
import numpy as np
import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim
import DTN
import time
import CWRU_Data

now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

# TODO continue train by fix lower layer and update adapatation layer !!!

MODEL_SAVE_PATH = "model/"
PRE_MODEL_NAME = "PRE_DTN_CWRU_Data.ckpt"

x_0HP, y_0HP = CWRU_Data.get_data('CWRU/0HP')
x_3HP, y_3HP = CWRU_Data.get_data('CWRU/3HP')
# train_X_3HP, test_X_3HP, train_y_3HP, test_y_3HP = train_test_split(x_3HP, y_3HP, test_size=0.3, random_state=0)
source_domain_data = CWRU_Data.Dataset(x_0HP, y_0HP)
target_domain_data = CWRU_Data.Dataset(x_3HP, y_3HP)

def con_MMD(xs, ys, xt, yt):
    mmd_mar = 4 * tf.reduce_mean(tf.multiply((tf.reduce_mean(xs, 0) - tf.reduce_mean(xt, 0)),
                                            (tf.reduce_mean(xs, 0) - tf.reduce_mean(xt, 0))))
    yss = tf.argmax(ys, axis=1)
    ytt = tf.argmax(yt, axis=1)
    mmd_con = 0
    for c in range(4):
        xs_c = tf.reshape(tf.boolean_mask(xs, tf.tile(tf.reshape(tf.equal(yss, c), [-1, 1]), [1, tf.shape(xs)[1]])), [-1, tf.shape(xs)[1]])  # TODO tile boolean mask to n_feature columns
        xt_c = tf.reshape(tf.boolean_mask(xt, tf.tile(tf.reshape(tf.equal(ytt, c), [-1, 1]), [1, tf.shape(xt)[1]])), [-1, tf.shape(xt)[1]])
        a = tf.reduce_mean(tf.multiply((tf.reduce_mean(xs_c, 0) - tf.reduce_mean(xt_c, 0)),
                                      (tf.reduce_mean(xs_c, 0) - tf.reduce_mean(xt_c, 0))))
        a = tf.where(tf.is_nan(a), 0., a)
        mmd_con = mmd_con + a
    return mmd_mar, mmd_con

with tf.Graph().as_default() as g:
    with tf.device('cpu:0'):

        xs = tf.placeholder(tf.float32, [source_domain_data.num_examples,
                                         DTN.IMAGE_SIZE,
                                         DTN.IMAGE_SIZE,
                                         DTN.NUM_CHANNELS],
                            name='source-x-input')

        xt = tf.placeholder(tf.float32, [target_domain_data.num_examples,
                                         DTN.IMAGE_SIZE,
                                         DTN.IMAGE_SIZE,
                                         DTN.NUM_CHANNELS],
                            name='target-x-input')

        ys_ = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='source-y-input')
        yt_ = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='source-y-input')


        # trainable=False fix lower layer param
        s_pool, s_fc1, s_fc2, ys = DTN.inference(xs, False, None, reuse=False, trainable=False)  # False: without dropout

        t_pool, t_fc1, t_fc2, yt = DTN.inference(xt, False, None, reuse=True, trainable=False)  # pesudo logit: yt

        # TODO ys_: ground truth label, yt: predicted label
        pool_mmd_mar, pool_mmd_con = con_MMD(s_pool, ys_, t_pool, yt)
        fc1_mmd_mar, fc1_mmd_con = con_MMD(s_fc1, ys_, t_fc1, yt)
        fc2_mmd_mar, fc2_mmd_con = con_MMD(s_fc2, ys_, t_fc2, yt)

        # TODO learning rate decay
        variables = slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if v.name.split('_')[0] != 'step']

        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            # print(variables_to_restore)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.getcwd() + '/model/%s' % PRE_MODEL_NAME)
            mar_pool, con_pool, mar_fc1, con_fc1, mar_fc2, con_fc2 = \
                sess.run([pool_mmd_mar, pool_mmd_con, fc1_mmd_mar, fc1_mmd_con, fc2_mmd_mar, fc2_mmd_con],
                         feed_dict={xs: np.reshape(source_domain_data.images,
                                                   (source_domain_data.num_examples,
                                                    DTN.IMAGE_SIZE,
                                                    DTN.IMAGE_SIZE,
                                                    DTN.NUM_CHANNELS)),
                                    ys_: source_domain_data.labels,
                                    xt: np.reshape(target_domain_data.images,
                                                   (target_domain_data.num_examples,
                                                    DTN.IMAGE_SIZE,
                                                    DTN.IMAGE_SIZE,
                                                    DTN.NUM_CHANNELS)),
                                    yt_: target_domain_data.labels})
            print('POOLING LAYER: MAR_MMD = %f, CON_MMD = %f' % (mar_pool, con_pool))
            print('FC1 LAYER: MAR_MMD = %f, CON_MMD = %f' % (mar_fc1, con_fc1))
            print('FC2 LAYER: MAR_MMD = %f, CON_MMD = %f' % (mar_fc2, con_fc2))

