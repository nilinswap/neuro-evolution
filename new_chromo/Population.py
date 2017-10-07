import numpy as np

class Population(object):

	def __init__(self, size, dim, max_hidden=20):
		self.size = size
		self.max_hidden = max_hidden
		self.input_dim = dim[0]
		self.output_dim = dim[1]

	def get_chromosome(self, hidden_nodes = -1):
		"""
		Create a random chromosome with given hidden nodes. 
		
		Parameters
		----------
		hidden_nodes : int
			No of hidden nodes. Randomly generated if -1.

		Returns
		-------
		chromo : dict
			Dictionary of all connection matrices. Attributes are hidden_nodes, w_input, w_hidden
		"""
		if hidden_nodes == -1:
			hidden_nodes = np.random.randint(self.max_hidden)
		max_x = (self.input_dim+1) * self.max_hidden
		max_y = (self.output_dim) * (self.max_hidden+1)
		x = (self.input_dim+1) * hidden_nodes
		y = (self.output_dim) * (hidden_nodes+1)
		
		#Creating empty matrix
		w_input = np.zeros((self.input_dim,self.max_hidden), dtype=np.float64)
		w_hidden = np.zeros((self.max_hidden, self.output_dim), dtype=np.float64)	

		for i in range(self.input_dim):
			for j in range(hidden_nodes):
				w_input[i,j] = np.random.uniform(-1,1)
	
		for i in range(hidden_nodes):
			for j in range(self.output_dim):
				w_hidden[i,j] = np.random.uniform(-1,1)

		chromo = {
			'hidden_nodes': hidden_nodes,
			'w_input': w_input,
			'w_hidden': w_hidden
		}

		return chromo

	def initial_pop(self):
		self.pop = []
		for i in range(self.size):
			self.pop.append(self.get_chromosome())

def main():	
	import copy
	dimtup=(8,1)
	population = Population(3,dimtup,5)
	population.initial_pop()
	print(population.pop)

if __name__=='__main__':
	main()
