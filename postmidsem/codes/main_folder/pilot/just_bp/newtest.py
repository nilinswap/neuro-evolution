import tensorflow as tf
import numpy  as np 

"""ap=tf.placeholder(dtype=tf.float64,shape=[None,1])

a=np.random.random((6,1))
half=tf.constant(0.5,dtype=ap.dtype)
dadum=tf.constant(0.5,dtype=ap.dtype)
q=tf.scan(lambda last,current: current[0],elems=ap,initializer=dadum)
s=tf.scan(lambda y,x: tf.greater_equal(x,half),elems=q,initializer=False)
print("hi",s)
r=tf.cast(s,dtype=tf.int32)

with tf.Session() as sess:
	print(sess.run([ap,r],feed_dict={ap:a}))
"""
#tf.add(tf.multiply(y[current],tf.log(self.p_y_given_x[current])),tf.multiply(tf.add(one,-y[current]),tf.log(tf.add(one,-self.p_y_given_x[current]))))
y=tf.placeholder(dtype=tf.int32,shape=[None,])

yai=tf.placeholder(dtype=tf.float64,shape=[None,1])

yval=np.random.randint(0,2,(5,))
yaival=np.random.rand(5,1)

dum=tf.constant(0.5,dtype=tf.float64)
minusone=tf.constant(-1,dtype=tf.int32)
one=tf.constant(1,dtype=y.dtype)
r=tf.scan(lambda last,current:last+1,elems=y,initializer=minusone)
w=tf.scan(lambda last,current: tf.add(tf.multiply(tf.cast(y[current],dtype=yai.dtype),tf.log(yai[current][0])),tf.multiply(tf.cast(tf.add(one,-y[current]),dtype=yai.dtype),tf.log(tf.add(tf.cast(one,dtype=yai.dtype),-yai[current][0])))),elems=r,initializer=dum)
z=-tf.reduce_mean(w)
with tf.Session() as sess:
	print(sess.run([w,r,yai,y,z],feed_dict={yai:yaival,y:yval}))