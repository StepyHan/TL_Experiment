import os
import logging
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import DTN
import time
import CWRU_Data

now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

BATCH_SIZE = 4000
LEARNING_RATE = 0.0005
REGULARZTION_RATE = 0

TRAINING_STEPS = 4000

theta1 = 10
theta2 = 10

# TODO continue train by fix lower layer and update adapatation layer !!!

MODEL_SAVE_PATH = "model/"
PRE_MODEL_NAME = "PRE_DTN_CWRU_Data.ckpt"
MODEL_NAME = "%s_reg-%s_t1-%s_t2-%s_NEWDTN_CWRU_Data.ckpt" % (now_time, REGULARZTION_RATE, theta1, theta2)

logfile = 'log/%s.log' % MODEL_NAME
logging.basicConfig(filename=logfile, level=logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

x_0HP, y_0HP = CWRU_Data.get_data('CWRU/0HP')
x_3HP, y_3HP = CWRU_Data.get_data('CWRU/3HP')
# train_X_3HP, test_X_3HP, train_y_3HP, test_y_3HP = train_test_split(x_3HP, y_3HP, test_size=0.3, random_state=0)
source_domain_data = CWRU_Data.Dataset(x_0HP, y_0HP)
target_domain_data = CWRU_Data.Dataset(x_3HP, y_3HP)
def new_regularizer(weights):
    return REGULARZTION_RATE * tf.reduce_mean(tf.exp(-tf.square(weights)))
    # return REGULARZTION_RATE * tf.reduce_mean(1 - tf.abs(weights))
def con_MMD(xs, ys, xt, yt):
    # TODO mask获得各类x
    # TODO 对x求MMD操作
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

logging.info('batch_size: %s, theta1: %s, theta2: %s, regular: %s' % (BATCH_SIZE, theta1, theta2, REGULARZTION_RATE))

with tf.Graph().as_default() as g:
    with tf.device('cpu:0'):
        X_TEST = tf.placeholder(tf.float32, [target_domain_data.num_examples,
                                              DTN.IMAGE_SIZE,
                                              DTN.IMAGE_SIZE,
                                              DTN.NUM_CHANNELS],
                                 name='test-x-input')
        Y_TEST = tf.placeholder(tf.float32, [None, DTN.OUTPUT_NODE], name='test-y-input')

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
        _, _, source_fc, ys = DTN.inference(xs, False, new_regularizer, reuse=False, trainable=False)  # False: without dropout

        _, _, target_fc, yt = DTN.inference(xt, False, new_regularizer, reuse=True, trainable=False)  # pesudo logit: yt




        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys, labels=tf.argmax(ys, 1))
        # 加上target y中有标签的部分
        cross_entropy_mean = tf.reduce_mean(cross_entropy)


        mmd_mar, mmd_con = con_MMD(source_fc, ys_, target_fc, yt)
        loss = cross_entropy_mean + theta1 * mmd_mar + theta2 * mmd_con + tf.add_n(tf.get_collection('losses'))

        global_ = tf.Variable(tf.constant(0), name='step_', trainable=False)
        lr = tf.train.exponential_decay(LEARNING_RATE, global_, decay_steps=100, decay_rate=0.99)
        train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # TODO ACCURACY CALCULATE
        _, _, _, Y_PREDICT = DTN.inference(X_TEST, False, None, reuse=True, trainable=False)
        correct_prediction = tf.equal(tf.argmax(Y_PREDICT, 1), tf.argmax(Y_TEST, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # TODO learning rate decay
        variables = slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if v.name.split('_')[0] != 'step']

        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            # print(variables_to_restore)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.getcwd() + '/model/%s' % PRE_MODEL_NAME)

            new_saver = tf.train.Saver()
            for i in range(TRAINING_STEPS):

                source_X, source_Y = source_domain_data.next_batch(BATCH_SIZE)
                target_X, target_Y = target_domain_data.next_batch(BATCH_SIZE)

                reshaped_xs = np.reshape(source_X, (BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))
                reshaped_xt = np.reshape(target_X, (BATCH_SIZE, DTN.IMAGE_SIZE, DTN.IMAGE_SIZE, DTN.NUM_CHANNELS))

                cross_entropy_loss, loss_value, _, MMD_MAR, MMD_CON = sess.run([cross_entropy_mean, loss, train, mmd_mar, mmd_con],
                                                                               feed_dict={xs: reshaped_xs,
                                                                                          ys_: source_Y,
                                                                                          xt: reshaped_xt
                                                                                          }
                                                                               )
                # print("yys:", yys)
                # print("yyt:", yyt)

                logging.info("After %d training steps, loss on training batch is %f, mmd_mar is %f, "
                             "mmd_con is %f, cross_entropy is %f" %
                             ((i + 1), loss_value, MMD_MAR, MMD_CON, cross_entropy_loss))
                if (i + 1) % 10 == 0:
                    accuracy_score = sess.run(accuracy, feed_dict={X_TEST: np.reshape(target_domain_data.images,
                                                                                      (target_domain_data.num_examples,
                                                                                       DTN.IMAGE_SIZE,
                                                                                       DTN.IMAGE_SIZE,
                                                                                       DTN.NUM_CHANNELS)),
                                                                   Y_TEST: target_domain_data.labels})
                    logging.info("After %d training steps, accuracy on target data: %f" % ((i+1), accuracy_score))
                if (i + 1) % 500 == 0:
                    new_saver.save(sess, os.path.join(MODEL_SAVE_PATH, '%s-%s' % (MODEL_NAME, (i+1))))


