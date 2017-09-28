import tensorflow as tf
import numpy  as np 
arrv=tf.placeholder(dtype=tf.float64,shape=[None,6])
y=tf.placeholder(dtype=tf.int32,shape=[None,])
i=tf.placeholder(dtype=tf.int32,shape=[None,])

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
#r=tf.concat((tf.reshape(q[0]),q[1]),axis=0)
with tf.Session() as sess:
	print(sess.run(z,feed_dict={y:yvalue}))
