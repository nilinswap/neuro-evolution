import tensorflow as tf
import numpy as np
with tf.variable_scope("model"):
	y=get_variable('y')
with tf.Session() as sess:
	print("here in test2.py",sess.run(y))
	