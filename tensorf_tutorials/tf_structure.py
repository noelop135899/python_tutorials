from __future__ import print_function
import tensorflow as tf
import  numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.33 + 0.75

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = x_data*Weights + biases


loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initializer_all_variable()    
else:
    init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step %20 == 0:
        print(step,sess.run(Weights),sess.run(biases))

