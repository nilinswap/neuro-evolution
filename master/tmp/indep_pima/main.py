import OptimizeNetwork
import GeneticFunctions
import network
import Population
import pimadataf

def main():
	rng=np.random.seed(1234)
	ONet = OptimizeNetwork.OptimizeNetwork(limit=500, switch_iter=200, prob_crossover=0.9, prob_mutation=0.2, scale_mutation=0.33333)

	popul = Population.Population(max_hidden_units=17, size=200, limittup=(-3,3))
	ONet.run(popul)
	print(ONet.best)

main()