import random

import numpy as np
import network
#import pimadataf
import chromosome
import tensorflow as tf
import time
import gene

import copy


def give_neg_log_likelihood(arr, oneDarr):
	parr = arr  # normalize(arr,axis = 0)
	if parr.shape[1] == 1:
		summer = np.sum([- (oneDarr[i] * np.log(parr[i, 0] + 0.000000000001) + (1 - oneDarr[i]) * np.log(
			1 - parr[i, 0] + 0.000000000001)) for i in range(parr.shape[0])])
	else:
		poneDarr = oneDarr.astype('int32')
		# print(oneDarr)

		summer = np.sum([- np.log(parr[i, poneDarr[i]] + 0.000000000001) for i in range(parr.shape[0])])
	return summer / parr.shape[0]


def give_mse(arr, oneDarr):
	onedarr = oneDarr.astype(dtype='int32')
	twodarr = np.zeros(arr.shape)
	for i in range(onedarr.shape[0]):
		twodarr[i][onedarr[i]] = 1
	return np.sum((arr - twodarr) ** 2)


def give_false_positive_ratio(arr, oneDarr):
	if arr.shape[1] > 2:
		print("false_positive is not appropriate objective, change objective function in Population.py")
		exit(1)
	if arr.shape[1] == 1:
		ar1 = np.where(arr > 0.5, 1, 0)
		ar1 = np.ravel(ar1)
	else:
		ar1 = np.argmax(arr, axis=1)
	summer = np.sum([ar1[i] * (1 - oneDarr[i]) for i in range(oneDarr.shape[0])])
	dummer = np.sum([(1 - ar1[i]) * (1 - oneDarr[i]) for i in range(ar1.shape[0])])
	return summer / (summer + dummer)


def give_false_negative_ratio(arr, oneDarr):
	if arr.shape[1] > 2:
		print("false_positive is not appropriate objective, change objective function in Population.py")
		exit(1)
	if arr.shape[1] == 1:
		ar1 = np.where(arr > 0.5, 1, 0)
		ar1 = np.ravel(ar1)
	else:
		ar1 = np.argmax(arr, axis=1)
	summer = np.sum([(1 - ar1[i]) * (oneDarr[i]) for i in range(oneDarr.shape[0])])
	dummer = np.sum([(ar1[i]) * (oneDarr[i]) for i in range(ar1.shape[0])])
	return summer / (summer + dummer)


def givesumar(size):
	ar = [0]
	for i in range(1, size + 1):
		ar += [ar[i - 1] + i]
	return ar


class Population(object):
	"""Class to create population object, and handle its methods"""

	def __init__(self, inputdim, outputdim, max_hidden_units, size=50, limittup=(-1, 1)):

		self.size = size
		self.max_hidden_units = max_hidden_units

		self.list_chromo = [chromosome.Chromosome(inputdim, outputdim) for i in range(self.size)]

		self.objective_arr = None

	def set_list_chromo(self, newlist_chromo):
		p = self.list_chromo
		self.list_chromo = newlist_chromo  # ndarray
		self.set_fitness()
		del (p)

	def set_objective_arr(self, network_obj):

		if not self.list_chromo:
			print("list_chromo is not set")
			exit(1)
		lis = []
		for chromo in self.list_chromo:
			# print("cmatrix",chromo.convert_to_MatEnc(network_obj.inputdim, network_obj.outputdim).CMatrix['IO'])
			outputarr = network_obj.feedforward_ne(chromo)
			# print(outputarr)
			# print(outputarr)

			# hot_vec = give_hot_vector( outputarr )

			neg_log_likelihood_val = give_neg_log_likelihood(outputarr, network_obj.resty)

			mean_square_error_val = give_mse(outputarr, network_obj.resty)

			false_positve_rat = give_false_positive_ratio(outputarr, network_obj.resty)

			false_negative_rat = give_false_negative_ratio(outputarr, network_obj.resty)

			lis.append([neg_log_likelihood_val, mean_square_error_val, false_positve_rat, false_negative_rat])
		self.objective_arr = np.array(lis)  # a 2d array of dimension #population X #objectives

	# print(self.objective_arr)

	def get_best(self):
		pass

	def get_average(self):
		pass

	def squa_test(x):
		return (x ** 2).sum(axis=1)


def main():
	import copy
	dimtup = (8, 1)
	pop = Population(4, dimtup, size=9)

	print(pop.list_chromo)

	neter = Network.Neterr(dimtup[0], dimtup[1], pop.list_chromo, pop.trainx, pop.trainy, pop.testx, pop.testy)


if __name__ == '__main__':
	main()
z = 0


def rand_init(inputdim, outputdim):
	global innov_ctr, z

	newchromo = chromosome.Chromosome(0)

	newchromo.node_ctr = inputdim + outputdim + 1
	innov_ctr = 1  # Warning!! these two lines change(reset) global variables, here might be some error
	lisI = [gene.Node(num_setter, 'I') for num_setter in range(1, newchromo.node_ctr - outputdim)]
	lisO = [gene.Node(num_setter, 'O') for num_setter in range(inputdim + 1, newchromo.node_ctr)]
	newchromo.node_arr = lisI + lisO
	for inputt in lisI:
		for outputt in lisO:
			newchromo.conn_arr.append(gene.Conn(innov_ctr, (inputt, outputt), z, status=True))
			z = z + 1
			innov_ctr += 1
	newchromo.bias_arr = [gene.BiasConn(outputt, random.random()) for outputt in lisO]
	newchromo.dob = 0
	return newchromo