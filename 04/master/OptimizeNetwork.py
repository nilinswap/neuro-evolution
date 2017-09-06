import GeneticFunctions
class OptimizeNetwork (GeneticFunctions):
	def __init__(self, D, limit=500, size=100, prob_crossover=0.9, prob_mutation=0.2,scale_mutation=0.33333):
	#	self.D = D
		self.counter = 0
		self.limit = limit
	#	self.size = size
		self.prob_crossover = prob_crossover
		self.prob_mutation = prob_mutation
		self.scale_mutation = scale_mutation
		self.best = (np.inf, []) #Add in class diagrams
		

	def probability_crossover(self):
		return self.prob_crossover

	def probability_mutation(self):
		return self.prob_mutation

	def crossover(self, parents):
		father, mother = parents
		
		child1 = []
		child2 = []
		alpha = random.uniform(0,1)
		for x in range(len(father)):
			child1.append(alpha*father[x] + (1-alpha)*mother[x])
			child2.append((1-alpha)*father[x] + alpha*mother[x])
		return (child1, child2)

	def selection(self, fitness_func):
		fits_pops=population.find_fitness(fitness_func)
		ranks = sorted(fits_pops, reverse = True)
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
	
	def terminate(self):
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