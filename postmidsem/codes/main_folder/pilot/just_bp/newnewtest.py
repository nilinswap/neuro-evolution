import tensorflow as tf
import numpy  as np 
arrv=tf.placeholder(dtype=tf.float64,shape=[None,6])
y=tf.placeholder(dtype=tf.int32,shape=[None,])
i=tf.placeholder(dtype=tf.int32,shape=[None,])
dum=tf.constant(0.5,dtype=tf.float64)
dadum=tf.constant(-1,dtype=tf.int32)
arrvalue=np.random.random((5,6))
yvalue=np.random.randint(0,6,(5,))
ivalue=np.arange(5)
print(arrvalue)
print(yvalue)
print(ivalue)
def func(last,current):
	return [last[0]+1,current]

q=tf.scan(fn=func,elems=y,initializer=[dadum,dadum])
z=tf.transpose(tf.stack([q[0],q[1]]))
print(z)
def newfunc(last,current):
	return current[1]+current[0]
w=tf.scan(fn=lambda last,current:tf.log(arrv[current[0]][current[1]]),elems=z,initializer=dum)
#r=tf.concat((tf.reshape(q[0]),q[1]),axis=0)
with tf.Session() as sess:
	print(sess.run(-tf.reduce_mean(w),feed_dict={arrv:arrvalue,y:yvalue}))
