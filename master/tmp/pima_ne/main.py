import OptimizeNetwork
import GeneticFunctions
import network
import Population
import pimadataf

def main():
	rest_set, test_set = pimadataf.give_data()

	ONet = OptimizeNetwork.OptimizeNetwork(limit=500, switch_iter=200,prob_crossover=0.9, prob_mutation=0.2,scale_mutation=0.33333)
	inputdim = 8
	pop_size = 100
	outputdim = 1
	
	for hid_nodes in range(0,17): #loop that changes hidden nodes 
		net = network.Network(inputdim, outputdim, hid_nodes, rest_set[0], rest_set[1], test_set[0], test_set[1])
		str_len = (inputdim+1) * hid_nodes + (hid_nodes+1) * outputdim
		popul = Population.Population(D=str_len, size=pop_size, net=net, limittup=(-1,1))
		print("here above run")
		ONet.run(popul)
	print(ONet.best)

main()
