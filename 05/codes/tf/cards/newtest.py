import tensorflow as tf
import numpy  as np 

ap=tf.placeholder(dtype=tf.float64,shape=[None,1])

a=np.random.random((6,1))
half=tf.constant(0.5,dtype=ap.dtype)
dadum=tf.constant(0.5,dtype=ap.dtype)
q=tf.scan(lambda last,current: current[0],elems=ap,initializer=dadum)
s=tf.scan(lambda y,x: tf.greater_equal(x,half),elems=q,initializer=False)
print("hi",s)
r=tf.cast(s,dtype=tf.int32)

with tf.Session() as sess:
	print(sess.run([ap,r],feed_dict={ap:a}))

