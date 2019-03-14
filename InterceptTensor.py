import tensorflow as tf

#Model parameters
w = tf.Variable([.4], dtype =tf.float32)
b = tf.Variable([-4], dtype = tf.float32)

x = tf.placeholder(tf.float32)

linear_model = w*x +b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:0}))