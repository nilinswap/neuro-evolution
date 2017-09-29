import tensorflow as tf
import numpy  as np 

y=tf.placeholder(dtype=tf.int32,shape=[None,])
yi=tf.placeholder(dtype=tf.int32,shape=[None,])
yar=np.random.randint(0,2,(7,))
yiar=np.random.randint(0,2,(7,))
print(yar,yiar)
r=tf.scan(lambda last,current:last+1,elems=y,initializer=-1)
qn=tf.scan(lambda last,current: tf.not_equal(y[current],yi[current]),elems=r,initializer=False)
q=tf.cast(qn,dtype=tf.int32)
#q=tf.scan(lambda last,current: (lambda i:),elems=qn,initializer=0)
#r=tf.scan((lambda last,current: current[1]),q)
#s=tf.reduce_mean(tf.cast(q,dtype=tf.float64))
#print(s)
with tf.Session() as sess:
	print(sess.run([q,yi,y],feed_dict={yi:yiar,y:yar}))
