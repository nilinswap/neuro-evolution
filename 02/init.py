def __init__(self, D, size,net,limittup=(-1,1)):
	self.D = D
	self.size = size
	self.net=net
	self.list_chromo = self.aux_pop(size,limittup, 16) 
	print(self.list_chromo)
	self.fits_pops=[]
	
def aux_pop(self, size,limittup, no_of_hidden_units):	
	population = []
	
	for i in range(1,no_of_hidden_units+1):
		for j in range(size//no_of_hidden_units):
			population.append(np.concatenate([[i],np.random.uniform(limittup[0],limittup[1],((self.net.inputdim+1)*i + (i+1)*self.net.outputdim))]))

	for i in range(1,size%no_of_hidden_units+1):
		population.append(np.concatenate([[i],np.random.uniform(limittup[0],limittup[1],((self.net.inputdim+1)*i + (i+1)*self.net.outputdim))]))
	
