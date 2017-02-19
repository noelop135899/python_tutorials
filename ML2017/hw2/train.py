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

'''
def compute_accuracy(acc):
    result = []
    dic = {0: 0, 1: 0}
    for i in range(len(acc)):
        if acc[i][0] > acc[i][1]:
            result.append(0)
        else:
            result.append(1)

    return result
'''
def read_line(file):
        X = []
        y = []
        for line in open(file, 'r'):
            row = line.rstrip('\r\n').split(',')
            for i in range(1, len(row)):
                row[i] = float(row[i])
            X.append(np.array(row[1:]))
            if row[-1] == 0:

                y.append(np.array([1, 0]))

            else:
                y.append(np.array([0, 1]))
        return X,y


def compute_accuracy(v_xs, v_ys):
    global prediction
    pre_ys = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(pre_ys,1), tf.argmax(v_ys,1))#數值最大的，是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={ys:v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 57])
ys = tf.placeholder(tf.float32, [None, 2])

train_X,train_y = read_line(sys.argv[1])
test_X,test_y = read_line(sys.argv[2])
train_X = np.delete(train_X,np.s_[57,57],axis=1)

#print(np.shape(train_y))
#print(np.shape(test_X))
#print(train_y)
#print(train_X[0],'\n',train_y[0])

pre = add_layer(xs, 57, 100, activation_function=tf.nn.sigmoid)
prediction = add_layer(pre, 100, 2, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
#saver = tf.train.Saver()
init = tf.global_variables_initializer()

sess.run(init)

for i in range(100):
    sess.run(train_step, feed_dict={xs: train_X, ys: train_y})
    if i % 100 == 0:
        #saver.save(sess, 'train_saver')  # 存取訓練數據
        #print('saver:',i)
        #accuracy = sess.run(prediction,feed_dict={xs: train_X})
        #result = compute_accuracy(accuracy)
        #print(compute_accuracy(train_X,train_y))# test 測試
        #print(compute_accuracy(test_X))# test 測試
        pass
#print(len(result))

