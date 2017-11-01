#gene.py

class Node:
	def __init__(self, node_ctr, nature):
		self.node_ctr = node_ctr
		self.nature = nature	#'I','H1','H2','O'

class Edge:
	def __init__(self, innov_ctr, couple, weight, status = True):
		self.innov_ctr = innov_ctr		#determines enabled or disabled
		self.source = couple[0]			#a node object
		self.destination = couple[1]	#same as above
		self,weight = weight
		self.status = enabled

	def get_status(self):
        return self.status

	def get_innov_num(self):
		return self.innov_ctr

	def get_weight(self):
		return self.weight

	def get_couple(self):
		return (self.source, self.destination)

class BiasEdge(Edge):
	def __init__(self, out_node, weight):
		node_ctr = -1
		nature = -1
		status = True
		Edge.__init__(-1, (Node(node_ctr, nature), out_node), weight, status)
