class GeneticFunctions(object):
	def probability_crossover(self):
		r"""returns rate of occur crossover(0.0-1.0)"""
		return 1.0

	def probability_mutation(self):
		r"""returns rate of occur mutation(0.0-1.0)"""
		return 0.0

	def fitness(self, chromosome):
		r"""returns domain fitness value of chromosome"""
		return len(chromosome)

	def terminate(self, fits_populations):
		r"""stop run if returns True
		- fits_populations: list of (fitness_value, chromosome)
		"""
		return False

	def selection(self, fits_populations):
		r"""generator of selected parents"""
		gen = iter(sorted(fits_populations))
		while True:
			f1, ch1 = next(gen)
			f2, ch2 = next(gen)
			yield (ch1, ch2)
			
		return

	def crossover(self, parents):
		r"""breed children"""
		return parents

	def mutation(self, chromosome):
		r"""mutate chromosome"""
		return chromosome
