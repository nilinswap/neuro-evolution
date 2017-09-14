import OptimizeNetwork
import GeneticFunctions
import network
import Population
import pimadataf

def main():
	#rest_set, test_set = pimadataf.give_data()

	ONet = OptimizeNetwork.OptimizeNetwork(limit=500, switch_iter=200,prob_crossover=0.9, prob_mutation=0.2,scale_mutation=0.33333)
	inputdim = 8
	pop_size = 200
	outputdim = 1
	max_no_of_hidden_units=17
	
	#net = network.Network(inputdim, outputdim, hid_nodes, rest_set[0], rest_set[1], test_set[0], test_set[1])
		
	popul = Population.Population(max_no_of_hidden_units=max_no_of_hidden_units,dimtup=(inputdim,outputdim), size=pop_size,  limittup=(-3,3))
	print("here above run")
	ONet.run(popul)
	print(ONet.best)

main()
