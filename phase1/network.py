
import numpy as np
#import tf_mlp
import tensorflow as tf
import time
import gene
import matenc
import chromosome
import pimadataf
import deep_net
from chromosome import *
import copy
import Population
'''
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
        rest_set, test_set = pimadataf.give_data()
        self.restx = rest_set[0]
        resty = rest_set[1]
        self.testx = test_set[0]
        testy = test_set[1]
        self.resty = np.ravel(resty)
        self.testy = np.ravel(testy)
        self.rest_setx = tf.Variable(initial_value = self.restx, name='rest_setx',
                                     dtype=tf.float32)
        self.rest_sety = tf.Variable(initial_value = self.resty, name='rest_sety', dtype=tf.int32)
        self.test_setx = tf.Variable(initial_value = self.testx, name='rest_sety',
                                     dtype=tf.float32)
        self.test_sety = tf.Variable(initial_value = self.testy, name='test_sety', dtype=tf.int32)


    def feedforward_cm(self, chromo, middle_activation = relu, final_activation = sigmoid,play = 0):

        self.inputarr = self.restx
        new_mat_enc = chromo.convert_to_MatEnc( self.inputdim, self.outputdim)

        input_till_H1 = middle_activation( np.dot( self.inputarr, new_mat_enc.CMatrix['IH1']*new_mat_enc.WMatrix['IH1'] ) )
        input_till_H2 = middle_activation( np.dot( input_till_H1, new_mat_enc.CMatrix['H1H2']*new_mat_enc.WMatrix['H1H2'] ) + np.dot( self.inputarr, new_mat_enc.CMatrix['IH2']*new_mat_enc.WMatrix['IH2'] ))
        bias_weight_arr = np.array([item.weight for item in new_mat_enc.Bias_conn_arr])
        output = final_activation( np.dot( input_till_H2, new_mat_enc.CMatrix['H2O']*new_mat_enc.WMatrix['H2O'] ) + np.dot( input_till_H1, new_mat_enc.CMatrix['H1O']*new_mat_enc.WMatrix['H1O'] ) + np.dot( self.inputarr, new_mat_enc.CMatrix['IO']*new_mat_enc.WMatrix['IO'] ) - bias_weight_arr )
        return output



        


    def feedforward_ne(self,chromosome,middle_activation=relu,final_activation=sigmoid, play = 0):


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
        self.inputarr = self.restx
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
        return final_activation(return_arr.reshape((self.inputarr.shape[0],self.outputdim)))       #a 2d matrix of dimension #datapoints X #outputdim


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
'''