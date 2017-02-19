import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 28, 28])
ys = tf.placeholder(tf.float32, [None, 10])

weight = {
    'in': tf.Variable(tf.random_normal([28, 128])),
    'out': tf.Variable(tf.random_normal([128, 10]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[128, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[10, ]))
}

def RNN(x, weight, biases):

    x = tf.reshape(x, [-1, 28])
    x_in = tf.matmul(x, weight['in']) +biases['in']
    x_in = tf.reshape(x_in, [-1, 28, 128])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(128, dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
    return tf.matmul(outputs[-1], weight['out']) + biases['out']

prediction = RNN(xs, weight, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    batch_xs,batch_ys = mnist.train.next_batch(128)
    batch_xs = batch_xs.reshape([128, 28, 28])
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys}))

