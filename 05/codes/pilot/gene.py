#gene.py

class Node:
	def __init__(self, node_num, nature):
		self.node_num = node_num
		self.nature = nature	#'I','H1','H2','O'

class Conn:
<<<<<<< HEAD
	def __init__(self, innov_num, couple, weight, status = True):
		self.innov_num = innov_num		#determines enabled or disabled
		self.source = couple[0]			#a node object
		self.destination = couple[1]	#same as above
		self,weight = weight
		self.status = status
=======
    def  __init__(self,innov_num,couple,weight,status = True):
        self.status=status          #determines enabled or disabled
        self.source=couple[0]          #a node object
        self.destination=couple[1]#same as above
        self.weight=weight
        self.innov_num=innov_num

    def get_couple(self):
        return (self.source,self.destination)
>>>>>>> 98264075887727120ec2815e454d62d7a04c24e8

	def get_status(self):
		return self.status

<<<<<<< HEAD
	def get_innov_num(self):
		return self.innov_num

	def get_weight(self):
		return self.weight
=======


class BiasConn(Conn):
    def __init__(self,out_node,weight):
        numb=-1
        nature=-1
        status = True
        Conn.__init__(-1,(Node(numb,nature),out_node),weight,status)
>>>>>>> 98264075887727120ec2815e454d62d7a04c24e8

	def get_couple(self):
		return (self.source, self.destination)

class BiasConn(Conn):
	def __init__(self, out_node, weight):
		node_num = -1
		nature = -1
		status = True
		Conn.__init__(-1, (Node(node_num, nature), out_node), weight, status)
