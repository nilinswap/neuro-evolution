import chromosome
import gene

def map_to_nodelis(maph):
    node_lis=[]
    for key in maph.keys():
        node_lis += [ gene.Node(x,key) for x in maph[key].values()]
    return node_lis

def split_key(st):
    if st == 'IH1':
        tup = ('I','H1')
    elif st == 'H1H2':
        tup = ('H1','H2')
    elif st == 'IH2':
        tup = ('I','H2')
    elif st == 'H2O':
        tup = ('H2','O')
    elif st == 'H1O':
        tup = ('H1','O')
    elif st == 'IO':
        tup = ('I','O')
    return tup

class MatEnc:
    def __init__(self,WMatrix,CMatrix,Bias_conn_arr,node_map, conn_map, node_lis, conn_lis):
        self.WMatrix = WMatrix            #both WMatrix and CMatrix are dictionary with keys - 'IO','IH1','H1,H2'...
        self.CMatrix = CMatrix            #same keys as above
        self.Bias_conn_arr = Bias_conn_arr
        self.node_map = node_map              #maps local index( at matrix) to node
        self.couple_to_conn_map = conn_map            #maps couple to innov num
        self.node_lis = node_lis #list of node objects
        self.conn_lis = conn_lis

    def convert_to_chromosome(self,inputdim,outputdim,dob):
        #newchromo.reset_chromo_to_zero()  # very important function, without it duplicate connections will be created
        dicW=self.WMatrix
        dicC=self.CMatrix
        #print([tup for tup in self.couple_map.items()][:3])
        for key in dicW.keys():
            key_tup = split_key(key)
            m,n = dicW[key].shape
            for row in range(m):
                for col in range(n):
                    if dicW[key][row][col]:

                        conn_here = None
                        node1 = self.node_map[key_tup[0]][row]
                        node2 = self.node_map[key_tup[1]][col]
                        couple = (node1, node2)
                        if couple in self.couple_to_conn_map:
                            conn_here = self.couple_to_conn_map[couple]
                        else:
                            print(row, col, key, "key_tup", key_tup)
                            print("key error h yaar")

                        #print ("hihihihi")
                        weight = dicW[key][row][col]
                        conn_here.weight = weight

        newchromo = chromosome.Chromosome(inputdim,outputdim)

        newchromo.reset_chromo_to_zero()

        newchromo.__setattr__('conn_arr', self.conn_lis)
        newchromo.__setattr__('bias_conn_arr', self.Bias_conn_arr)
        newchromo.__setattr__('node_arr', self.node_lis)
        newchromo.__setattr__('dob', dob)
        newchromo.set_node_ctr()
        return newchromo