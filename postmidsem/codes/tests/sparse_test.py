import numpy as np
import tensorflow as tf
import time

st = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=[30, 40])

mat = tf.placeholder( shape = (40,30), dtype = tf.float32)

npmat =  np.zeros((40,30))
npmat[0][0]=2.1
npmat[1][1]=3.2
npmat[30][23]=4.5
newmat =  tf.matmul( tf.sparse_to_dense(st.indices, st.dense_shape, st.values), mat )
newnewmat= tf.sparse_matmul( tf.sparse_to_dense(st.indices, st.dense_shape, st.values), mat, a_is_sparse = True, b_is_sparse = True)

with tf.Session() as sess:
	t1=time.time()
	mate = newmat.eval( feed_dict = {mat : npmat})
	t2 = time.time()
	print(t2-t1,mate)
	t3 = time.time()
	matee = newmat.eval( feed_dict = { mat: npmat})
	t4 = time.time()
	print(t4-t3, matee)

print("done")	
	
