import random
import numpy as np
from irisnew import give_data
from irisnew import find_fitness

rest_setx, rest_sety = give_data()[0]
test_setx, test_sety = give_data()[1]

class GeneticAlgorithm(object):
	def __init__(self, genetics):
		self.genetics = genetics

	def run(self):
		population = self.genetics.initial()
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
		self.D = D
		self.counter = 0
		self.limit = limit
		self.size = size
		self.prob_crossover = prob_crossover
		self.prob_mutation = prob_mutation
		self.best = (np.inf, [])
		
	def probability_crossover(self):
		return self.prob_crossover

	def probability_mutation(self):
		return self.prob_mutation

	def initial(self):
		return [self.random_chromo() for j in range(self.size)]

	#>>>>>>======== Put your fitness function here ===========<<<<<<
	def fitness(self, chromosome):
		pass

	def check_stop(self, fits_populations):
		self.counter += 1
		f = sorted(fits_populations, reverse = True)

		if f[-1][0] < self.best[0]:
			self.best = f[-1]

		if self.counter < 300:
			self.prob_mutation = 0.2
		else:
			self.prob_mutation = 0.02

		if self.counter % 10 == 0:	
			fits = [f for f, ch in fits_populations]
			best = min(fits)
			ave = sum(fits) / len(fits)
			print(
				"[G %3d] score=(%.4f, %.4f)" %
				(self.counter, best, ave))

		if self.counter >= self.limit:
			#print("Best fitness achieved: " + str(self.best))
			#print(type(self.best[1]))
			print(find_fitness(test_setx, test_sety, self.best[1]))
			return True
		return False

	def crossover(self, parents, method=1):
		father, mother = parents
		if method == 1:
			child1 = []
			child2 = []
			alpha = random.uniform(0,1)
			for x in range(len(father)):
				child1.append(alpha*father[x] + (1-alpha)*mother[x])
				child2.append((1-alpha)*father[x] + alpha*mother[x])

		elif method == 2:
			index1 = random.randint(1, len(father) - 2)
			index2 = random.randint(1, len(father) - 2)
			if index1 > index2: index1, index2 = index2, index1
			child1 = father[:index1] + mother[index1:index2] + father[index2:]
			child2 = mother[:index1] + father[index1:index2] + mother[index2:]
		
		return (child1, child2)

	def parents(self, fits_populations):
		ranks = sorted(fits_populations, reverse = True)
		rank_array = []
		for i in range(len(ranks)):
			for x in range(i+1):
				rank_array.append(ranks[i][1])

		father = rank_array[random.randint(0, len(rank_array)-1)]
		mother = rank_array[random.randint(0, len(rank_array)-1)]
		return (father, mother)

	def mutation(self, chromosome):
		mutated = chromosome
		for x in range(len(chromosome)):
			if random.random() < self.prob_mutation:
				vary = np.random.randn(1,1)/3
				mutated[x] += vary
		return mutated

	def random_chromo(self):
		return [random.uniform(-1,1) for i in range(self.D)]
	
GeneticAlgorithm(OptimizeFunction(15)).run()
