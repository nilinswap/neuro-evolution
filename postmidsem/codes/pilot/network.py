
import numpy as np
#import tf_mlp
#import tensorflow as tf
import time
import gene
import matenc
import chromosome

from chromosome import *

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))
def relu(arr):
    return np.where(arr>0,arr,0)
def priortize_connections(conn_lis):
    dict={'IH1':[],
          'H1H2':[],
          'IH2':[],
          'H2O':[],
          'H1O':[],
          'IO':[]
          }
    for concsn in conn_lis:
        tup=concsn.get_couple()
        dict[tup[0].nature+tup[1].nature].append(concsn)
    return dict['IH1']+['breakH1']+dict['H1H2']+dict['IH2']+['breakH2']+dict['H2O']+dict['H1O']+dict['IO']
class Neterr:
    def __init__(self, inputdim, outputdim,inputarr,  hidden_unit_lim ,rng):
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.inputarr=inputarr  #self explanatory
        self.hidden_unit_lim = hidden_unit_lim
        self.rng = rng

        #self.arr_of_net = arr_of_net
    """
    def set_arr_of_net(self, newarr_of_net):
        self.arr_of_net = newarr_of_net
    """
    def convert_chromo_to_MatEnc(self,chromo):

        ConnMatrix = {}  # Connection Matrix
        WeightMatrix = {}  # Weight Matrix
        NatureCtrDict = {}  # Contains Counter of Nature { 'I', 'H1', 'H2', 'O' }
        NatureCtrDict['I'] = 0
        NatureCtrDict['H1'] = 0
        NatureCtrDict['H2'] = 0
        NatureCtrDict['O'] = 0

        dictionary = {}  # Contains node numbers mapping starting from 0, nature-wise
        dictionary['I'] = {}
        dictionary['H1'] = {}
        dictionary['H2'] = {}
        dictionary['O'] = {}
        couple_to_innov_map={}



        for i in chromo.node_arr:
            dictionary[i.nature][i] = NatureCtrDict[i.nature]
            NatureCtrDict[i.nature] += 1

        ConnMatrix['IO'] = np.zeros((self.inputdim, self.outputdim))
        ConnMatrix['IH1'] = np.zeros((self.inputdim, NatureCtrDict['H1']))
        ConnMatrix['IH2'] = np.zeros((self.inputdim, NatureCtrDict['H2']))
        ConnMatrix['H1H2'] = np.zeros((NatureCtrDict['H1'], NatureCtrDict['H2']))
        ConnMatrix['H1O'] = np.zeros((NatureCtrDict['H1'], self.outputdim))
        ConnMatrix['H2O'] = np.zeros((NatureCtrDict['H2'], self.outputdim))

        WeightMatrix['IO'] = np.zeros((self.inputdim, self.outputdim))
        WeightMatrix['IH1'] = np.zeros((self.inputdim, NatureCtrDict['H1']))
        WeightMatrix['IH2'] = np.zeros((self.inputdim, NatureCtrDict['H2']))
        WeightMatrix['H1H2'] = np.zeros((NatureCtrDict['H1'], NatureCtrDict['H2']))
        WeightMatrix['H1O'] = np.zeros((NatureCtrDict['H1'], self.outputdim))
        WeightMatrix['H2O'] = np.zeros((NatureCtrDict['H2'], self.outputdim))

        for con in chromo.conn_arr:
            if con.status == True:
                ConnMatrix[con.source.nature + con.destination.nature][
                    dictionary[con.source.nature][con.source]][
                    dictionary[con.destination.nature][con.destination]] = 1
            couple_to_innov_map[con.get_couple()]=con.innov_num
            WeightMatrix[con.source.nature + con.destination.nature][
                dictionary[con.source.nature][con.source]][
                dictionary[con.destination.nature][con.destination]] = con.weight

        inv_dic = { key : { v : k for k,v in dictionary[key].items() } for key in dictionary.keys()}

        new_encoding = matenc.MatEnc( WeightMatrix, ConnMatrix, chromo.bias_conn_arr, inv_dic, couple_to_innov_map, chromo.node_arr)

        return new_encoding

    def feedforward_cm(self, chromo, middle_activation = relu, final_activation = sigmoid,play = 0):
        if play:
            maten = self.convert_chromo_to_MatEnc(chromo)
            chromo = maten.convert_to_chromosome(chromo.dob)
            print("here")

        new_mat_enc = self.convert_chromo_to_MatEnc( chromo)
        if 1:
            print("right here right now")
            chromo = new_mat_enc.convert_to_chromosome(chromo.dob)
            new_mat_enc = self.convert_chromo_to_MatEnc(chromo)
            print("right here")
        input_till_H1 = middle_activation( np.dot( self.inputarr, new_mat_enc.CMatrix['IH1']*new_mat_enc.WMatrix['IH1'] ) )
        input_till_H2 = middle_activation( np.dot( input_till_H1, new_mat_enc.CMatrix['H1H2']*new_mat_enc.WMatrix['H1H2'] ) + np.dot( self.inputarr, new_mat_enc.CMatrix['IH2']*new_mat_enc.WMatrix['IH2'] ))
        bias_weight_arr = np.array([item.weight for item in new_mat_enc.Bias_conn_arr])
        output = final_activation( np.dot( input_till_H2, new_mat_enc.CMatrix['H2O']*new_mat_enc.WMatrix['H2O'] ) + np.dot( input_till_H1, new_mat_enc.CMatrix['H1O']*new_mat_enc.WMatrix['H1O'] ) + np.dot( self.inputarr, new_mat_enc.CMatrix['IO']*new_mat_enc.WMatrix['IO'] ) - bias_weight_arr )
        return output



        


    def feedforward_ne(self,chromosome,middle_activation=relu,final_activation=sigmoid, play = 0):
        if play:
            maten = self.convert_chromo_to_MatEnc(chromosome)
            chromosome = maten.convert_to_chromosome(chromosome.dob)
            print("here")

        print("inside feedforward")

        conn_list = priortize_connections(
            chromosome.conn_arr)  # list of connections with string type breaks to seperate
        #[item.pp() for item in conn_list if type(item) != str]
        """for item in conn_list :
            if type(item) != str:
                item.pp()
            else:
                print(item)
        """
        return_arr = np.array([])
        for i in range(self.inputarr.shape[0]):

            storage = [0.0 for i in range(self.hidden_unit_lim + self.outputdim)]   #wtf!! there was an error here because I wrote 0 instead of 0.0!
            storage = np.array([0.0]+list(self.inputarr[i])+storage) #here [0] is dummy storage as we use '1' indexing for node_ctr

            node_num_lis=[]
            for connection in conn_list:

                if type(connection)==str:
                    #print("before",storage)
                    for node_num in node_num_lis:
                        storage[node_num]=middle_activation(storage[node_num])
                    node_num_lis=[]
                    #print("after",storage)
                    continue

                tup = connection.get_couple()
                node_num_lis.append(tup[1].node_num)
                weight = connection.__getattribute__('weight')
                if connection.status == True:
                    #connection.pp()
                    #print(storage[tup[1].node_num], storage[tup[0].node_num]*weight)
                    #print(
                    storage[tup[1].node_num] += storage[tup[0].node_num]*weight
                    #print(storage)
            #print(storage)

            bias_weights=[bn.weight for bn in chromosome.bias_conn_arr]
            for p in range(len(bias_weights)):
                storage[self.inputdim + 1+p]    += -1*bias_weights[p]
            output_part = storage[self.inputdim+1:self.outputdim+self.inputdim+1]
            return_arr = np.concatenate((return_arr,output_part))
        return final_activation(return_arr.reshape((self.inputarr.shape[0],self.outputdim)))


        #pass


    def test(self, weight_arr):
        pass

    def modify_thru_backprop(self, popul, epochs=10, learning_rate=0.01, L1_reg=0.00001, L2_reg=0.0001):
        pass


def squa_test(x):
    return (x ** 2).sum(axis=1)

def dummy_popultation(number):#return list of chromosomes
    chromolis=[]
    for i in range(number):
        newchromo=chromosome.Chromosome(0)
        newchromo.rand_init()
        chromolis.append(newchromo)
    return chromolis



def main1():
    indim = 4
    outdim = 1
    arr_of_net = np.zeros((4,4))
    np.random.seed(4)
    neter = Neterr(indim, outdim, arr_of_net, 10, np.random)
    neter.feedforwardcm(indim, outdim, arr_of_net, np.random)

def test1():
    for_node = [(i,'I') for i in range(1,4)]
    for_node += [(i,'O') for i in range(4,6)]
    for_node += [(i,'H2') for i in range(6,8)]
    node_ctr=8
    innov_num=11
    dob=0
    node_lis = [gene.Node(x,y) for x,y in for_node]
    for_conn = [ (1,(1,4),0.4,False), (2,(1,5),0.25,True), (3,(2,4),0.25,True), (4,(2,5),0.5,True), (5,(3,4),0.7,True),
                 (6,(3,5),0.6,False), (7,(1,6),0.5,True), (8,(6,4),0.4,True), (9,(3,7),0.3,True), (10,(7,5),0.6,True)
                 ]
    conn_lis = [gene.Conn(x,(node_lis[tup[0]-1],node_lis[tup[1]-1]),w,status) for x,tup,w,status in for_conn]
    for_bias=[ (4,0.2),(5,0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x-1],y) for x,y in for_bias]

    newchromo = Chromosome(dob, node_lis, conn_lis, bias_conn_lis)
    newchromo.set_node_ctr(node_ctr)
    #newchromo.pp()
    def calc_output_directly(inputarr):
        lis=[]
        for arr in inputarr:
            output1 = sigmoid( relu(arr[0]*0.5)*0.4 + 0.25*arr[1] + 0.7*arr[2] - 0.2 )
            output2 = sigmoid( arr[0]*0.25 + arr[1]*0.5 + relu(arr[2]*0.3)*0.6 - 0.1 )
            lis.append([output1,output2])
        return np.array(lis)

    inputarr=np.array([[3,2,1],[4,1,2]])
    indim = 3
    outdim = 2
    np.random.seed(4)
    num_data = 2
    #inputarr = np.random.random((num_data, indim))
    neter = Neterr(indim, outdim, inputarr, 10, np.random)
    print(neter.feedforward_ne(newchromo))
    print(calc_output_directly(inputarr))
    print(neter.feedforward_cm(newchromo))
    #print(neter.feedforward_ne(newchromo, play =1))


def test2():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st='2212211'
    for_node +=  [(i+6,'H'+st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (19, (2, 12), 0.6, True), (20, (12, 7), 0.4, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True), (24, (11, 5), 0.25, True),
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo = Chromosome(dob, node_lis, conn_lis, bias_conn_lis)
    newchromo.set_node_ctr(node_ctr)

    # newchromo.pp()
    def calc_output_directly(inputarr):
        lis = []
        for arr in inputarr:
            x1 = arr[0]
            x2 = arr[1]
            x3 = arr[2]
            output1 = sigmoid(
                                0.3 * x1     +
                                0.1 * relu(
                                            0.7 * relu(0.5 * x1) +
                                            0.2 * x1
                                        )    +
                                0.15 * relu(
                                            0.1 * x2 +
                                            0.4 * relu(
                                                0.6 * x2 +
                                                0.8 * x3
                                            )
                                        )     +
                                0.75 * relu(
                                            0.6 * x2 +
                                            0.8 * x3
                                        )     -
                                 0.2
                        )
            #output2 = sigmoid(arr[0] * 0.25 + arr[1] * 0.5 + relu(arr[2] * 0.3) * 0.6 - 0.1)
            output2 = sigmoid(
                0.5 * x3 +
                1 * relu(
                    0.15 * relu(0.25 * x1) +
                    0.9 * x2
                ) +
                0.25 * relu(
                    0.25 * x1
                ) +
                0.77 * relu(

                    0.33 * x3
                ) -
                0.1
            )
            lis.append([output1, output2])
        return np.array(lis)

    inputarr = np.array([[0, 2, 1], [0.8, 1, 2]])
    indim = 3
    outdim = 2
    np.random.seed(4)
    num_data = 2
    # inputarr = np.random.random((num_data, indim))
    neter = Neterr(indim, outdim, inputarr, 10, np.random)
    print(neter.feedforward_ne(newchromo))
    print(calc_output_directly(inputarr))
    tempchromo = newchromo
    print(neter.feedforward_cm(newchromo))
    if (newchromo == tempchromo) :
        print("yeah they are equal")
    print(neter.feedforward_ne(newchromo, play=1))
    print(neter.feedforward_cm(newchromo, play=1))
    print ( "done")

    def interchanging_test( chromo ):
        new_mat_enc = neter.convert_chromo_to_MatEnc(chromo)
        newchromo = new_mat_enc.convert_to_chromosome(0)
        if newchromo.bias_conn_arr != chromo.bias_conn_arr:
            print("falied 1")
        if newchromo.node_arr != chromo.node_arr:
            print("failed 2")
        if newchromo.dob != chromo.dob or newchromo.node_ctr != chromo.node_ctr:
            print("failed 3", "node_ctr are", newchromo.node_ctr, chromo.node_ctr, len(newchromo.node_arr), len(chromo.node_arr))
        listup1 = [(con.status,con.weight,con.get_couple(),con.innov_num) for con in newchromo.conn_arr]
        listup2 = [(con.status, con.weight, con.get_couple(), con.innov_num) for con in chromo.conn_arr]
        #print( set(listup1),set(listup2))
        if set(listup1) != set(listup2):
            print("failed 4")
    interchanging_test(newchromo)


def main():
    indim=4
    outdim=3
    np.random.seed(4)
    num_data=2
    inputarr=np.random.random((num_data,indim))
    neter = Neterr(indim, outdim, inputarr, 10, np.random)
    chromo=chromosome.Chromosome(0)
    chromo.rand_init(indim,outdim,np.random)
    print(neter.feedforward_ne(chromo))

    print(neter.feedforward_cm(chromo))


if __name__ == '__main__':
    test2()

