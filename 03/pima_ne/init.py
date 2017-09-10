def __init__(self, D, size,net,no_of_hidden_layers,limittup=(-1,1)):
	self.D = D
	self.size = size
	self.net=net
	self.fits_pops=[]
	#self.list_chromo = np.random.uniform(limittup[0],limittup[1],(self.size,(self.net.inputdim+1)*self.net.hid_nodes+(self.net.hid_nodes+1)*self.net.outputdim))
	self.list_chromo = aux_pop(size,net,no_of_hidden_layers)

def aux_pop(self, size,net, no_of_hidden_layers):	
	population = []
	for i in range(no_of_hidden_layers):
		if i%2 == 0:
			for j in range(12):
				population[i] = np.concatenate([i],np.random.uniform(limittup[0],limittup[1],((self.net.inputdim+1)*i + (i+1)*self.net.outputdim)))
		else:
			for j in range(13):
				population[i] = np.concatenate([i],np.random.uniform(limittup[0],limittup[1],((self.net.inputdim+1)*i + (i+1)*self.net.outputdim)))

	return population