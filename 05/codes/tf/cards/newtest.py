import tensorflow as tf
import numpy  as np 
arrv=tf.placeholder(dtype=tf.float64,shape=[None,6])
y=tf.placeholder(dtype=tf.int32,shape=[None,])
i=tf.placeholder(dtype=tf.int32,shape=[None,])
dum=tf.constant(5.0,dtype=tf.float64)
dadum=tf.constant(0,dtype=tf.int32)
arrvalue=np.random.random((5,6))
yvalue=np.random.randint(0,6,(5,))
ivalue=np.arange(5)
print(arrvalue)
print(yvalue)
print(ivalue)
def func(last,current):
	return tf.log(arrv[current][y[current]])

q=tf.scan(fn=func,elems=i,initializer=dum)

with tf.Session() as sess:
	print(sess.run(q,feed_dict={arrv:arrvalue,y:yvalue,i:ivalue}))
