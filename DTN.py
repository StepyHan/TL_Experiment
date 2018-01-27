#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
DTN底层由LeNet构成
'''
import tensorflow as tf

# 定义神经网络相关的参数
INPUT_NODE = None
OUTPUT_NODE = 4
LAYER1_NODE = 500

IMAGE_SIZE = 32
NUM_CHANNELS = 1
NUM_LABELS = 4

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 3
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3

CONV3_DEEP = 128
CONV3_SIZE = 3

CONV4_DEEP = 256
CONV4_SIZE = 3
# 全连接层的节点个数
FC1_SIZE = 500
FC2_SIZE = 500

def inference(input_tensor, train, regularizer, reuse, trainable):
    with tf.variable_scope('layer1-conv1', reuse=reuse):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        trainable=trainable)
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0),
                                       trainable=trainable)
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-pool', reuse=reuse):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2', reuse=reuse):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       trainable=trainable)
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0),
                                       trainable=trainable)
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer4-pool2', reuse=reuse):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #
    # # Add 3rd con-layer with 128 channels and kernel size is 16x16
    # with tf.variable_scope('layer5-conv3', reuse=reuse):
    #     conv3_weight = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
    #                                    initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                                    trainable=trainable)
    #     conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0),
    #                                    trainable=trainable)
    #     conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
    #     relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    #
    # with tf.variable_scope('layer6-pool3', reuse=reuse):
    #     pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # # Add 4th con-layer with 256 channels and kernel size is 8x8
    # with tf.variable_scope('layer7-conv4', reuse=reuse):
    #     conv4_weight = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
    #                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
    #     conv4 = tf.nn.conv2d(pool3, conv4_weight, strides=[1, 1, 1, 1], padding='SAME')
    #     relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    # with tf.variable_scope('layer8-pool4', reuse=reuse):
    #     pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # Full connected layer
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]   # pool_shape[0]是一个batch中样本的数目 其余的代表体积
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1', reuse=reuse):
        fc1_weights = tf.get_variable("weight", [nodes, FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      trainable=trainable)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC1_SIZE], initializer=tf.constant_initializer(0.1),
                                     trainable=trainable)
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2', reuse=reuse):
        fc2_weights = tf.get_variable("weight", [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)


    with tf.variable_scope('layer7-fc3', reuse=reuse):
        fc3_weights = tf.get_variable('weight', [FC2_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return reshaped, fc1, fc2, logit

