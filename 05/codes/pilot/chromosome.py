#chromosome.py

import gene
node_ctr = None
innov_ctr = None
class Chromosome:
    def __init__(self,dob,node_arr=[],conn_arr=[],bias_arr=[]):
        self.node_arr=node_arr#list of node objects
        self.conn_arr=conn_arr#list of conn objects
        self.bias_arr=bias_arr#list of BiasNode objects
        self.dob=dob # the generation in which it was created.
    def rand_init(self,inputdim,outputdim,rng):
        global node_ctr
        global innov_ctr
        node_ctr=inputdim+outputdim+1
        innov_ctr = 0
        lisI = [gene.Node(num_setter,'I') for num_setter in range(1,node_ctr-outputdim)]
        lisO =  [gene.Node(num_setter,'O') for num_setter in range(inputdim+1,node_ctr)]
        self.node_arr = lisI + lisO
        for inputt in lisI:
            for outputt in lisO:
                innov_ctr+=1
                self.conn_arr.append( gene.Conn(innov_ctr, inputt, outputt, rng.random(), status=True))
        self.bias_arr=[gene.BiasConn(outputt,rng.random()) for outputt in lisO]
        self.dob=0