from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
#from theano.ifelse import ifelse
import numpy as np
import tensorflow as tf
#import theano
#import theano.tensor as T
import tf_load_data
import random
import tf_hiddenlayer
from tf_mlp import MLP

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='cards.data', n_par=10, n_hidden=100,freq_par=0.1):
	rng = np.random
	#savo=None          #so that 'if' block sees the global savo

	nin = 32
	nout = 5

	lis = tf_load_data.give_target_data()

	rest_set=lis[0]#tuple of two shared variable of array
	test_set=lis[1]#tuple of shared variable of array
	rest_setx = tf.Variable(initial_value=rest_set[0], name='rest_setx', dtype=tf.float64)
	rest_sety = tf.Variable(initial_value=rest_set[1], name='rest_sety', dtype=tf.int32)
	test_setx = tf.Variable(initial_value=test_set[0], name='rest_sety', dtype=tf.float64)
	test_sety = tf.Variable(initial_value=test_set[1], name='test_sety', dtype=tf.int32)
	#var_lis=[rest_setx,rest_sety,test_setx,test_sety]
	#nodelis=[rxn,ryn,txn,tyn]
	#savo1=tf.train.Saver(var_list=[rest_setx,rest_sety,test_setx,test_sety])
	par_size=int(rest_setx.shape[0])//n_par
	#print(rest_set[0].get_value().shape[0])
	#
	prmsdind=tf.placeholder('int32')
	
	
	valid_x_to_be=rest_setx[prmsdind*par_size:(prmsdind+1)*par_size,:]
	valid_y_to_be=rest_sety[prmsdind*par_size:(prmsdind+1)*par_size]
	train_x_to_be=tf.concat((rest_setx[:(prmsdind)*par_size,:],rest_setx[(prmsdind+1)*par_size:,:]),axis=0)
	train_y_to_be=tf.concat((rest_sety[:(prmsdind)*par_size],rest_sety[(prmsdind+1)*par_size:]),axis=0)
	# allocate symbolic variables for the data
		#index = T.lscalar()  # index to a [mini]batch
	x = tf.placeholder(dtype=tf.float64,name='x',shape=[None,nin]) 
	y = tf.placeholder(dtype=tf.int32,name='y',shape=[None,])

	##yen_hidden=T.lscalar()
	n_hid=5
	classifier= MLP(
		    rng=rng,
		    input=x,
		    n_in=nin,
		    n_hidden=n_hid,
		    n_out=nout		#this was important, I tried taking 1 gave fatal results, decided to fix this later.
		)

	cost = tf.add(tf.add(
	    classifier.negative_log_likelihood(y)
	    ,L1_reg * classifier.L1
	    ),L2_reg * classifier.L2_sqr
	)

	print(cost)

	optmzr = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	#optmzr = tf.train.AdamOptimizer().minimize(cost,var_list=[valid_x_to_be,valid_y_to_be,train_x_to_be,train_y_to_be])
	
	zhero=0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#savo1.restore(sess, "/home/placements2018/forgit/neuro-evolution/05/state/tf/cards/model.ckpt")
		sess.run([classifier.logRegressionLayer.W.initializer,classifier.logRegressionLayer.b.initializer,classifier.hiddenLayer.W.initializer,classifier.hiddenLayer.b.initializer])
		#print("------------------------------------------------------------------")
		#print(sess.run([valid_x_to_be,valid_y_to_be,train_x_to_be,train_y_to_be],feed_dict={prmsdind:0}))
		
		print(sess.run([cost],feed_dict={x:train_x_to_be.eval(feed_dict={prmsdind:zhero}),y:train_y_to_be.eval(feed_dict={prmsdind:zhero})}))
		
		#cool thing starts from here ->
	    ######################
	    # BUILD ACTUAL MODEL #
	    ######################

		print('...building the model')

	
		for epoch in range(n_epochs):
			for ind in range(n_par):
				_, bost = sess.run([optmzr,cost], feed_dict={x:train_x_to_be.eval(feed_dict={prmsdind:ind}),y:train_y_to_be.eval(feed_dict={prmsdind:ind})})
			print( bost )
			if epoch%10==0:

				print(sess.run([classifier.errors(y)],feed_dict={x:valid_x_to_be.eval(feed_dict={prmsdind:ind}),y:valid_y_to_be.eval(feed_dict={prmsdind:ind})}))
		st = str(sess.run(classifier.errors(y),feed_dict={x:test_setx.eval(),y:test_sety.eval()}))
		fileo = open("./log_folder/log_bp_"+str(n_hidden)+"_"+str(n_epochs)+".txt", "a")
		fileo.write(st+'\n')
		fileo.close()
		print("testing",st)

def main():
	for i in range(int(sys.argv[1])):
 		test_mlp(n_hidden=5,n_epochs=400)
if __name__ == '__main__':
    main()

