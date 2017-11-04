import tensorflow as tf
import numpy  as np 
arrv=tf.placeholder(dtype=tf.float64,shape=[None,6])
y=tf.placeholder(dtype=tf.int32,shape=[None,])
dum=tf.constant(5.0,dtype=tf.float64)
arrvalue=np.random.random((5,6))
yvalue=np.arange(5)

def func(last,current):
	return arrv[current][current]

q=tf.scan(fn=func,elems=y,initializer=dum)

with tf.Session() as sess:
	print(sess.run(q,feed_dict={arrv:arrvalue,y:yvalue}))
