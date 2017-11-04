#chromosome.py

import gene

innov_ctr = None

class Chromosome:

    def __init__(self,dob,node_arr=[],conn_arr=[],bias_arr=[]):
        self.node_arr = node_arr	#list of node objects
        self.conn_arr = conn_arr	#list of conn objects
        self.bias_conn_arr = bias_arr	#list of BiasNode objects
        self.dob = dob 				#the generation in which it was created.
        self.node_ctr=len(node_arr)
    def rand_init(self,inputdim,outputdim,rng):

        global innov_ctr
        self.node_ctr = inputdim+outputdim+1
        innov_ctr = 1                                   # Warning!! these two lines change(reset) global variables, here might be some error
        lisI = [gene.Node(num_setter,'I') for num_setter in range(1,self.node_ctr-outputdim)]
        lisO =  [gene.Node(num_setter,'O') for num_setter in range(inputdim+1,self.node_ctr)]
        self.node_arr = lisI + lisO
        for inputt in lisI:
            for outputt in lisO:

                self.conn_arr.append( gene.Conn(innov_ctr, (inputt, outputt), rng.random(), status=True))
                innov_ctr += 1
        self.bias_arr = [gene.BiasConn(outputt,rng.random()) for outputt in lisO]
        self.dob = 0
    def set_node_ctr(self,ctr):
        self.node_ctr = ctr
    def pp(self):

        print("\nNode List")
        [item.pp() for item in self.node_arr]

        print("\n\nConnection List")
        [item.pp() for item in self.conn_arr]

        print("\n\nBias Connection List")
        [item.pp() for item in self.bias_conn_arr]

        print("dob",self.dob,"node counter",self.node_ctr)
        print("--------------------------------------------")




