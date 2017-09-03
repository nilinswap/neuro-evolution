import random
import numpy as np
from irisnew import give_data
from irisnew import find_fitness

class GeneticAlgorithm(object):
	def __init__(self, genetics):
		self.genetics = genetics

	def run(self):
		population = self.genetics.initial()
		rest_setx, rest_sety = give_data()[0]
		test_setx, test_sety = give_data()[1]
		while True:
			#Calculate fitness for each chromosome in population
			fits_pops = [(find_fitness(rest_setx, rest_sety, ch),  ch) for ch in population]
			if self.genetics.check_stop(fits_pops):
				break
			population = self.next(fits_pops)
		return population

	def next(self, fits):
		size = len(fits)
		nexts = []
		while len(nexts) < size:
			parents = self.genetics.parents(fits)
			cross = random.random() < self.genetics.probability_crossover()
			children = self.genetics.crossover(parents) if cross else parents
			for ch in children:
				nexts.append(self.genetics.mutation(ch))
		return nexts[0:size]
	

"""
example: Mapped guess prepared Text
"""
class OptimizeFunction():
	def __init__(self, D, limit=500, size=100, prob_crossover=0.9, prob_mutation=0.2):
		self.counter = 0
		self.limit = limit
		self.D = D
		self.size = size
		self.prob_crossover = prob_crossover
		self.prob_mutation = prob_mutation
		
	# GeneticFunctions interface impls
	def probability_crossover(self):
		return self.prob_crossover

	def probability_mutation(self):
		return self.prob_mutation

	def initial(self):
		return [self.random_chromo() for j in range(self.size)]

	#>>>>>>======== Put your fitness function here ===========<<<<<<
	def check_stop(self, fits_populations):
		self.counter += 1
		f = sorted(fits_populations, reverse = True)

		if self.counter < 100:
			self.prob_mutation = 0.2
		else:
			self.prob_mutation = 0.02

		if self.counter % 10 == 0:	
			fits = [f for f, ch in fits_populations]
			best = min(fits)
			worst = max(fits)
			ave = sum(fits) / len(fits)
			print(
				"[G %3d] score=(%.4f, %.4f, %.4f): %f" %
				(self.counter, best, ave, worst, f[0][0]))

		if self.counter == 500:
			print(fits_populations)
		return self.counter >= self.limit

	def crossover(self, parents, method=1):
		father, mother = parents
		if method == 1:
			child1 = []
			child2 = []
			alpha = random.uniform(0,1)
			for x in range(len(father)):
				child1.append(alpha*father[0] + (1-alpha)*mother[0])
				child2.append((1-alpha)*father[x] + alpha*mother[x])

		elif method == 2:
			index1 = random.randint(1, len(self.target) - 2)
			index2 = random.randint(1, len(self.target) - 2)
			if index1 > index2: index1, index2 = index2, index1
			child1 = father[:index1] + mother[index1:index2] + father[index2:]
			child2 = mother[:index1] + father[index1:index2] + mother[index2:]
		
		return (child1, child2)

	def parents(self, fits_populations, method=2):
		if method == 1:
			father = self.tournament(fits_populations)
			mother = self.tournament(fits_populations)
			return (father, mother)

		elif method == 2:
			ranks = sorted(fits_populations, reverse = True)
			rank_array = []
			for i in range(len(fits_populations)):
				for x in range(i+1):
					rank_array.append(fits_populations[i][1])

			father = rank_array[random.randint(0, len(rank_array)-1)]
			mother = rank_array[random.randint(0, len(rank_array)-1)]
			return (father, mother)

	def tournament(self, fits_populations):
		alicef, alice = self.select_random(fits_populations)
		bobf, bob = self.select_random(fits_populations)
		return alice if alicef > bobf else bob

	def mutation(self, chromosome):
		mutated = chromosome
		for x in range(len(chromosome)):
			if random.random() < self.prob_mutation:
				vary = np.random.randn(1,1)/3
				mutated[x] += vary
		return mutated

	def select_random(self, fits_populations):
		return fits_populations[random.randint(0, len(fits_populations)-1)]

	def random_chromo(self):
		return [random.uniform(-1,1) for i in range(self.D)]
	
GeneticAlgorithm(OptimizeFunction(15)).run()
