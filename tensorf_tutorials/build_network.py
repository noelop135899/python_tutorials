from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input,Weight) + biases
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)

x_data = np.linspace(-1, 1, 300)[ :, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 +noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data,color = 'r')
plt.ion()
plt.show()

for i in range(500):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i % 25 == 0:
        try:
            ax.lines.remove(line[0])
        except:
            pass
        line = ax.plot(x_data, sess.run(prediction,feed_dict={xs:x_data,ys:y_data}),
                    'b--',lw=5)
        plt.pause(0.5)
    