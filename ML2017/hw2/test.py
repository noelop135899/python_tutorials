import tensorflow as tf
import numpy as np
import sys


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

xs = tf.placeholder(tf.float32, [None, 57])
ys = tf.placeholder(tf.float32, [None, 2])

X = []
y = []
for line in open(sys.argv[1],'r'):
    row = line.rstrip('\r\n').split(',')
    for i in range(1, len(row)):
        row[i] = float(row[i])
    X.append(np.array(row[1:-1]))
    if row[-1] == 0:
        y.append(np.array([1, 0]))

    else:
        y.append(np.array([0, 1]))

tX = []
ty = []
for line in open(sys.argv[2],'r'):
    trow = line.rstrip('\r\n').split(',')
    for i in range(1, len(trow)):
        trow[i] = float(trow[i])
    tX.append(np.array(trow[1:-1]))
    if trow[-1] == 0:
        ty.append(np.array([1, 0]))

    else:
        ty.append(np.array([0, 1]))


pre = add_layer(xs, 57, 100, activation_function=tf.nn.sigmoid)
prediction = add_layer(pre, 100, 2, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'train_saver')

for i in range(100):
    sess.run(train_step, feed_dict={xs: X, ys: y})
    if i % 20 == 0:
        print(compute_accuracy(tX, ty))

