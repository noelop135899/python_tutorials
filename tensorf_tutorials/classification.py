from __future__ import print_function
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None,):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biaese = tf.Variable(tf.zeros([1,out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs,Weight) + biaese
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)
def compute_accuracy(v_xs, v_ys):
    global prediction
    pre_ys = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(pre_ys,1), tf.argmax(v_ys,1))#數值最大的，是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={ys:v_ys})
    return result


xs = tf.placeholder(tf.float32, [None,784])
ys = tf.placeholder(tf.float32, [None,10])

prediction = add_layer(xs, 784, 10 ,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                reduction_indices=[1]))#loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs : batch_xs, ys : batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))# test 測試

