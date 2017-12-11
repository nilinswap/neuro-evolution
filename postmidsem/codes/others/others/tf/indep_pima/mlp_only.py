import numpy as np
# import theano
import tf_mlp
import tensorflow as tf
import time


def sigmoid(arr):
	return 1 / (1 + np.exp(-arr))


class Neterr:
	def __init__(self, inputdim, outputdim,  rest_setx, rest_sety, test_setx, test_sety, rng, n_par=10):
		"""trainx=trainarr[:,:inputdim]
		trainy=trainarr[:,inputdim:]
		testx=testarr[:,:inputdim]
		testy=testarr[:,inputdim:]
		"""
		self.inputdim = inputdim
		self.outputdim = outputdim
		self.srest_setx = rest_setx
		self.srest_sety = rest_sety
		self.stest_setx = test_setx
		self.stest_sety = test_sety
		self.rng = rng

		self.x = tf.placeholder(name='x', dtype=tf.float64, shape=[None, self.inputdim])
		self.y = tf.placeholder(name='y', dtype=tf.int32, shape=[None, ])

		savo1 = tf.train.Saver(var_list=[self.srest_setx, self.srest_sety, self.stest_setx, self.stest_sety])
		with tf.Session() as sess:
			savo1.restore(sess,
						  "/home/iit2015087/forgit/neuro-evolution/postmidsem/others/state/tf/indep_pima/input/model.ckpt")  # only restored for this session
			self.trainx = self.srest_setx.eval()
			self.trainy = self.srest_sety.eval()
			self.testx = self.stest_setx.eval()
			self.testy = self.stest_sety.eval()
		self.n_par = n_par
		par_size = int(self.trainx.shape[0] / n_par)
		self.prmsdind = tf.placeholder(name='prmsdind', dtype=tf.int32)
		self.valid_x_to_be = self.srest_setx[self.prmsdind * par_size:(self.prmsdind + 1) * par_size, :]
		self.valid_y_to_be = self.srest_sety[self.prmsdind * par_size:(self.prmsdind + 1) * par_size]
		self.train_x_to_be = tf.concat(
			(self.srest_setx[:(self.prmsdind) * par_size, :], self.srest_setx[(self.prmsdind + 1) * par_size:, :]),
			axis=0)
		self.train_y_to_be = tf.concat(
			(self.srest_sety[:(self.prmsdind) * par_size], self.srest_sety[(self.prmsdind + 1) * par_size:]), axis=0)



	#

	def test(self, weight_arr):
		hid_nodes = int(weight_arr[0])
		fir_weight = weight_arr[1:(self.inputdim + 1) * hid_nodes + 1].reshape(self.inputdim + 1, hid_nodes)
		sec_weight = weight_arr[(self.inputdim + 1) * hid_nodes + 1:].reshape((hid_nodes + 1), self.outputdim)
		testx = np.concatenate((self.testx, -np.ones((self.testx.shape[0], 1))), axis=1)
		midout = np.dot(testx, fir_weight)
		midout = np.tanh(midout)
		midout = np.concatenate((midout, -np.ones((midout.shape[0], 1))), axis=1)
		output = np.dot(midout, sec_weight)
		output = sigmoid(output)
		for i in range(len(output)):
			if output[i] > 0.5:
				output[i] = 1
			else:
				output[i] = 0
		er_arr = np.mean(abs(output - self.testy))
		# er_arr = (1/2)*np.mean((output-self.testy)**2)

		return er_arr

	def modify_thru_backprop(self,epochs=10, learning_rate=0.01, L1_reg=0.00001, L2_reg=0.0001):

		lis = []
		hid_nodes = 100


		newmlp = tf_mlp.MLP(self.x, self.inputdim, self.outputdim, hid_nodes, self.rng)


		fullnet = newmlp

		cost = tf.add(tf.add(
			fullnet.negative_log_likelihood(self.y)
			, L1_reg * fullnet.L1
		), L2_reg * fullnet.L2_sqr
		)

		optmzr = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
		savo1 = tf.train.Saver(var_list=[self.srest_setx, self.srest_sety, self.stest_setx, self.stest_sety])
		with tf.Session() as sess:

			savo1.restore(sess, "/home/iit2015087/forgit/neuro-evolution/postmidsem/others/state/tf/indep_pima/input/model.ckpt")
			sess.run([fullnet.logRegressionLayer.W.initializer, fullnet.logRegressionLayer.b.initializer,
					  fullnet.hiddenLayer.W.initializer, fullnet.hiddenLayer.b.initializer])
			# print("------------------------------------------------------------------")
			# print(sess.run([valid_x_to_be,valid_y_to_be,train_x_to_be,train_y_to_be],feed_dict={self.prmsdind:0}))

			# print(sess.run([cost],feed_dict={x:train_x_to_be.eval(feed_dict={self.prmsdind:zhero}),y:train_y_to_be.eval(feed_dict={self.prmsdind:zhero})}))

			# cool thing starts from here ->
			######################
			# BUILD ACTUAL MODEL #
			######################

			print('...building the model')
			print("nhid", hid_nodes)





			err = sess.run(fullnet.errors(self.y), feed_dict={self.x: self.trainx, self.y: self.trainy})
			print("train error ", err)

			#print("feedforward err", popul.fits_pops[pind])

			# just any no. which does not satisfy below condition
			prev = 7
			current = 5
			start1 = time.time()
			for epoch in range(epochs):
				listisi = []
				for ind in range(self.n_par):
					_, bost = sess.run([optmzr, cost], feed_dict={
						self.x: self.train_x_to_be.eval(feed_dict={self.prmsdind: ind}),
						self.y: self.train_y_to_be.eval(feed_dict={self.prmsdind: ind})})

					if epoch % (epochs // 4) == 0:
						q = fullnet.errors(self.y).eval(
							feed_dict={self.x: self.valid_x_to_be.eval(feed_dict={self.prmsdind: ind}),
									   self.y: self.valid_y_to_be.eval(feed_dict={self.prmsdind: ind})})
						listisi.append(q)
				if epoch % (epochs // 4) == 0:
					prev = current
					current = np.mean(listisi)
					print('validation', current)

				if prev - current < 0.002:
					break;
			end1 = time.time()
			print("time ", end1 - start1)

			print("testing error", self.test(fullnet.turn_weights_into_chromosome()))



# popul.set_fitness()


def squa_test(x):
	return (x ** 2).sum(axis=1)


def main():
	pass



if __name__ == '__main__':
	main()
