from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import cifar10_input as cf10

flags = tf.app.flags
flags.DEFINE_string('train_dir', 'cifar10_train','Directory where to write event logs and checkpoint.')
FLAGS = flags.FLAGS

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def train():
    images, labels = cf10.distort_input()
    #Tensor("shuffle_batch_1:0", shape=(128, 24, 24, 3), dtype=float32)
    #Tensor("Reshape_3:0", shape=(128,), dtype=int32)

    ## conv1 layer ##
    W_conv1 = weight_variable([5, 5, 3, 64])  # patch 5x5, in size 3, out size 64
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)  # output size 24x24x64
    h_pool1 = max_pool_2x2(h_conv1)  # output size 12x12x64

    ## conv2 layer ##
    W_conv2 = weight_variable([5,5, 64, 128]) # patch 5x5, in size 64, out size 128
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 12x12x128
    h_pool2 = max_pool_2x2(h_conv2)  # output size 6x6x128

    ## fc1 layer ##
    W_fc1 = weight_variable([6 * 6 * 128, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 6, 6, 128] ->> [n_samples, 6*6*128]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    ## fc2 layer ##
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    #Tensor("Softmax:0", shape=(128, 10), dtype=float32)

    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=prediction,))

    train_step = tf.train.AdagradOptimizer(5e-3).minimize(cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step)
        if i % 50 == 0:
            print('done')


def main(_):
    cf10.download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()