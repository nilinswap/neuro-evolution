import numpy as np

class Population(object):
	"""Class to create population object, and handle its methods"""
	list_chromo = []
	fits_pops = []

	def __init__(self, D, size):
		self.D = D
		self.size = size
		self.list_chromo = np.array([self.random_chromo() for j in range(self.size)])

	def find_fitness(self, fitness_func):
		self.fits_pops = [(fitness_func(ch), ch) for ch in self.list_chromo]

	def random_chromo(self):
		return [random.uniform(-1,1) for i in range(self.D)]

	def get_best(self):
		return sorted(self.fits_pops, reverse = True)[-1]

	def get_average(self):
		return sum(self.fits_pops[0])/self.size
