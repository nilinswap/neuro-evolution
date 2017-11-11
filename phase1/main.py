
import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
from deap import tools

from Population import Population
from network import Neterr
'''
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))

toolbox = base.Toolbox()
class Individual():
	pass

def zdt1(individual):
	return f1, f2, f3, f4


toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
'''
def main(seed=None):
	#random.seed(seed)
	print("hi")
	indim = 8
	outdim = 1
	n_hidden = 10
	size = 4
	#pop = Population.Population(indim,outdim,n_hidden,size), 
	#net = network.Neterr(indim, outdim, popo.list_chromo, n_hidden, np.random)
	#pop.set_objective_arr(net)
	print("hi")
	#print(size(pop.list_chromo[0].node_arr))

"""
def main2(seed=None):
	random.seed(seed)

	NGEN = 250
	MU = 100
	CXPB = 0.9

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	# stats.register("avg", numpy.mean, axis=0)
	# stats.register("std", numpy.std, axis=0)
	stats.register("min", numpy.min, axis=0)
	stats.register("max", numpy.max, axis=0)
	
	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"
	indim = 8
	outdim = 1
	n_hidden = 10
	size = 4
	pop = Population.Population(indim,outdim,n_hidden,size), 
	net = network.Neterr(indim, outdim, popo.list_chromo, n_hidden, np.random)
	pop.set_objective_arr(net)
	print(pop)
	#pop = toolbox.population(n=MU)

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in pop if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	# This is just to assign the crowding distance to the individuals
	# no actual selection is done
	pop = toolbox.select(pop, len(pop))
	
	record = stats.compile(pop)
	logbook.record(gen=0, evals=len(invalid_ind), **record)
	print(logbook.stream)

	# Begin the generational process
	for gen in range(1, NGEN):
		# Vary the population
		offspring = tools.selTournamentDCD(pop, len(pop))
		offspring = [toolbox.clone(ind) for ind in offspring]
		
		for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
			if random.random() <= CXPB:
				toolbox.mate(ind1, ind2)
			
			toolbox.mutate(ind1)
			toolbox.mutate(ind2)
			del ind1.fitness.values, ind2.fitness.values
		
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# Select the next generation population
		pop = toolbox.select(pop + offspring, MU)
		record = stats.compile(pop)
		logbook.record(gen=gen, evals=len(invalid_ind), **record)
		print(logbook.stream)

	#print("Final population hypervolume is %f" % HyperVolume(pop, [11.0, 11.0]))

	return pop, logbook
		
if __name__ == "__main__":
	with open("zdt1_front.json") as optimal_front_data:
		 optimal_front = json.load(optimal_front_data)
	#Use 500 of the 1000 points in the json file
	optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
	
	pop, stats = main()
	pop.sort(key=lambda x: x.fitness.values)
	
	'''print(stats)
	print("Convergence: ", convergence(pop, optimal_front))
	print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))'''
	
	import matplotlib.pyplot as plt
	import numpy
	
	front = numpy.array([ind.fitness.values for ind in pop])
	optimal_front = numpy.array(optimal_front)
	plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
	plt.scatter(front[:,0], front[:,1], c="b")
	plt.axis("tight")
	plt.show()
	"""

if __name__ == "__main__":
	main()