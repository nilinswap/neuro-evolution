import numpy as np
import tensorflow as tf

def dense_to_sparse(mat):
	idx = tf.where( tf.not_equal( mat, 0.0 ) )
	sparse = tf.SparseTensor( idx, tf.gather_nd( mat, idx), mat.get_shape)
	
	return sparse

def test():
	mat = tf.placeholder( shape = (2,3), dtype = tf.float32)
	npmat = np.zeros((2,3))
	npmat[0,1] = npmat[1,2] = 2.0
	with tf.Session() as sess:
		print(sess.run(dense_to_sparse(mat), feed_dict = { mat : npmat }))

test()	
