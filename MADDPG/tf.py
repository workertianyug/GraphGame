import tensorflow as tf


a = tf.constant([5,5,5],tf.float32)

b = tf.placeholder(tf.float32,shape=[3])

c = tf.add(a,b)

d = tf.stack([c,c,c])

init_op = tf.global_variables_initializer()


g = tf.constant([[1],[2],[3]])

# with tf.Session() as sess:
# 	cv,dv = sess.run([c,d],feed_dict={b:[1,1,1]})

g1 = tf.slice(g,[0,0],[1,1])
g2 = tf.slice(g,[1,0],[1,1])
g3 = tf.slice(g,[2,0],[1,1])

# with tf.Session() as sess:
# 	gg,gg1,gg2,gg3 = sess.run([g,g1,g2,g3])


k = tf.placeholder(tf.float32, shape=[1,None])

k3 = tf.slice(k,[0,0],[1,3])

with tf.Session() as sess:

	kk,kk3 = sess.run([k,k3],feed_dict = {k:[[0,1,2]]})






