import chromosome
import gene
def map_to_nodelis(maph):
    node_lis=[]
    for key in maph.keys():
        node_lis += [ gene.Node(x,key) for x in maph[key].values()]
    return node_lis

class MatEnc:
    def __init__(self,WMatrix,CMatrix,Bias_conn_arr,node_num_map,node_lis):
        self.WMatrix = WMatrix            #both WMatrix and CMatrix are dictionary with keys - 'IO','IH1','H1,H2'...
        self.CMatrix = CMatrix            #same keys as above
        self.Bias_conn_arr = Bias_conn_arr
        self.node_num_map = node_num_map
        self.node_lis = node_lis  #list of node objects


    def covert_to_chromosome(self,dob):
        newchromo = chromosome.Chromosome(dob)
        # map_to_lis(self.node_num_map)
        newchromo.node_arr = self.node_lis
        newchromo.bias_conn_arr = self.Bias_conn_arr
        newchromo.node_ctr = len(newchromo.node_arr)






