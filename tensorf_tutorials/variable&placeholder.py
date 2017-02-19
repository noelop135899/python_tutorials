import tensorflow as tf

state = tf.Variable(5,name='counter')
#print(state)#name/read/shape/dtrpe
one = tf.Variable(10)

value_add = tf.add(state, one) #value＿add指是一個行為，執行時並不會加到state裡
update = tf.assign(state, value_add) #加載

if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


"""method : placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
ouput = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
#[14.]

"""