import OptimizeNetwork
import GeneticFunctions
import Network
import Population
import pimadataf
import numpy as np

def main():
	rng=np.random
	#rng.seed(100)
	ONet = OptimizeNetwork.OptimizeNetwork(rng,limit=500, switch_iter=200, prob_crossover=0.9, prob_mutation=0.2, scale_mutation=0.33333)
	popul = Population.Population(rng,max_hidden_units=17, size=6, limittup=(-3,3))
	ONet.run(popul)

main()