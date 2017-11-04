import tensorflow as tf
import numpy as np
"""
with tf.variable_scope("model"):
	y=tf.Variable(np.random.random((4,)),name='y',dtype=tf.float64)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("here in test1.py",sess.run(y))
with tf.variable_scope("model",reuse=True):
	z=tf.get_variable('y')
with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	print("here in test1.py again",sess.run(y))"""
y=tf.Variable(np.random.random((3,)),name='y',dtype=tf.float32)

z=None
with tf.Session() as sess:
	sess.run(y.initializer)
	z=y.eval()

print(z)