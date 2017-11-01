#chromosome.py

import gene

class Chromosome:
    def __init__(self,dob):
        self.node_arr = []	#list of node objects
        self.conn_arr = []	#list of conn objects
        self.bias_arr = []	#list of BiasNode objects
        self.dob = dob 		# the generation in which it was created.
