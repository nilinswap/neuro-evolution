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
    def __init__(self,WMatrix,CMatrix,Bias_conn_arr,node_map, conn_map, node_lis):
        self.WMatrix = WMatrix            #both WMatrix and CMatrix are dictionary with keys - 'IO','IH1','H1,H2'...
        self.CMatrix = CMatrix            #same keys as above
        self.Bias_conn_arr = Bias_conn_arr
        self.node_map = node_map              #maps local index( at matrix) to node
        self.couple_map = conn_map            #maps couple to innov num
        self.node_lis = node_lis                #list of node objects


    def convert_to_chromosome(self,indim,outdim,dob):
        newchromo = chromosome.Chromosome(indim,outdim)
        newchromo.reset_chromo_to_zero()  # very important function, without it duplicate connections will be created
        newchromo.dob = dob
        # map_to_lis(self.node_num_map)
        newchromo.node_arr = self.node_lis
        newchromo.bias_conn_arr = self.Bias_conn_arr
        newchromo.set_node_ctr()
        dicW=self.WMatrix
        dicC=self.CMatrix
        #print([tup for tup in self.couple_map.items()][:3])
        for key in dicW.keys():
            key_tup = split_key(key)
            m,n = dicW[key].shape
            for row in range(m):
                for col in range(n):
                    if dicW[key][row][col]:
                        if dicC[key][row][col]:
                            status = True
                        else:
                            status = False

                        node1 = self.node_map[key_tup[0]][row]
                        node2 = self.node_map[key_tup[1]][col]
                        couple = (node1,node2)
                        if couple in self.couple_map:
                            innov_num = self.couple_map[couple]
                        else:
                            print(row,col,key,"key_tup",key_tup)
                            print("key error h yaar")

                        #print ("hihihihi")
                        weight = dicW[key][row][col]
                        new_conn=gene.Conn(innov_num,couple,weight,status)
                        newchromo.conn_arr.append(new_conn)

        return newchromo









