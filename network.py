
import numpy as np
#import tf_mlp
import tensorflow as tf
import time
import gene
import matenc
import chromosome
import dataset2_dataf
from chromosome import *
import copy

import dataset3_dataf
import pickle
def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))
def relu(arr):
    return np.where(arr>0,arr,0)
def softmax(arr):
    assert(arr.shape[1] > 1)
    comp_wise_exp = np.exp(arr)
    return comp_wise_exp/comp_wise_exp.sum(axis = 1).reshape((arr.shape[0],1))
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
    def __init__(self, inputdim, outputdim,   hidden_unit_lim , change_to_target , rng = random):# HYPERPARAMETER hid_unit_lim
        self.inputdim = inputdim
        self.outputdim = outputdim
        #self.inputarr=inputarr  #self explanatory
        self.hidden_unit_lim = hidden_unit_lim
        self.rng = rng
        #rest_set, test_set = pimadataf.give_data()#a two tuple of ( two tuple of array)
        if not change_to_target:
            rest_set, test_set = dataset2_dataf.give_source_data()  # a two tuple of ( two tuple of array)
        elif change_to_target == 1:
            rest_set, test_set = dataset2_dataf.give_target_data()
        elif change_to_target == 2:
            rest_set, test_set = dataset2_dataf.give_target_data_just_src_just_tar()
        elif change_to_target == 100:

            with open("./pickle_jar/src_tup.pickle", "rb") as fp:
                rest_set, _ = pickle.load(fp)
                #print(rest_set[0].shape, rest_set[1].shape)
            with open("./pickle_jar/tar_tup.pickle", "rb") as fp:
                trest_set, ttest_set = pickle.load(fp)
                #print(trest_set[0].shape, trest_set[1].shape, ttest_set[0].shape, ttest_set[1].shape)
            with open("./pickle_jar/dublue.pickle", "rb") as fp:
                W_mat = pickle.load(fp)

            rest_set = np.concatenate((rest_set[0], np.transpose(np.dot(W_mat, np.transpose(trest_set[0]))))), np.concatenate((rest_set[1], trest_set[1]))

            #print("here ", rest_set[0].shape, rest_set[1].shape)
            test_set = (np.transpose( np.dot(W_mat, np.transpose(ttest_set[0]))), ttest_set[1])
            #print("here ", test_set[0].shape, test_set[1].shape)
        # FOR ANY CHANGE IN DATASET, CHANGE DIMENSION NO. MENTIONED IN THESE THREE FILES - cluster.py, chromosome.py and main_just_tar.py
        self.restx = rest_set[0]
        resty = rest_set[1]
        self.testx = test_set[0]
        testy = test_set[1]
        #print("one time", resty.shape, testy.shape, self.restx.shape)

        self.resty = np.ravel(resty)
        self.testy = np.ravel(testy)
        self.rest_setx = tf.Variable(initial_value = self.restx, name='rest_setx',
                                     dtype=tf.float32)
        self.rest_sety = tf.Variable(initial_value = self.resty, name='rest_sety', dtype=tf.int32)
        self.test_setx = tf.Variable(initial_value = self.testx, name='rest_sety',
                                     dtype=tf.float32)
        self.test_sety = tf.Variable(initial_value = self.testy, name='test_sety', dtype=tf.int32)
        #self.inputarr = inputarr
        self.inputarr = self.restx
        #print("shape here",self.restx.shape)

    def feedforward_cm(self, chromo, middle_activation = relu, final_activation = sigmoid,play = 0):


        new_mat_enc = chromo.convert_to_MatEnc( self.inputdim, self.outputdim)

        input_till_H1 = middle_activation( np.dot( self.inputarr, new_mat_enc.CMatrix['IH1']*new_mat_enc.WMatrix['IH1'] ) )
        input_till_H2 = middle_activation( np.dot( input_till_H1, new_mat_enc.CMatrix['H1H2']*new_mat_enc.WMatrix['H1H2'] ) + np.dot( self.inputarr, new_mat_enc.CMatrix['IH2']*new_mat_enc.WMatrix['IH2'] ))
        bias_weight_arr = np.array([item.weight for item in new_mat_enc.Bias_conn_arr])
        output = final_activation( np.dot( input_till_H2, new_mat_enc.CMatrix['H2O']*new_mat_enc.WMatrix['H2O'] ) + np.dot( input_till_H1, new_mat_enc.CMatrix['H1O']*new_mat_enc.WMatrix['H1O'] ) + np.dot( self.inputarr, new_mat_enc.CMatrix['IO']*new_mat_enc.WMatrix['IO'] ) - bias_weight_arr )
        return output



        


    def feedforward_ne(self,chromosome,middle_activation=relu,final_activation=sigmoid, play = 0):


        #print("inside feedforward")

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
        return final_activation(return_arr.reshape((self.inputarr.shape[0],self.outputdim)))       #a 2d matrix of dimension #datapoints X #outputdim


        #pass



    def test_err(self, chromo):
        temp = self.inputarr
        self.inputarr = self.testx
        arr = self.feedforward_ne(chromo)
        if arr.shape[1] == 1:
            newar = np.where(arr > 0.5, 1, 0)
            newar = np.ravel(newar)
        else:
            newar = np.argmax(arr, axis=1)
        newarr = np.where(newar != self.testy, 1, 0)
        #print(newarr)
        self.inputarr = temp
        return np.mean(newarr)

    def test_on_pareto_patch(self,pareto_set):
        temp = self.inputarr
        self.inputarr = self.testx
        ctr =0
        lis = []
        minh = 1000000
        for chromo in pareto_set:
            arr = self.feedforward_ne(chromo)
            if arr.shape[1] == 1:
                newar = np.where(arr > 0.5, 1, 0)
                newar = np.ravel(newar)
            else:
                newar = np.argmax(arr, axis=1)
            newarr = np.where(newar != self.testy, 1, 0)
            lis += list(newarr)

            tempo = minh
            minh = min(minh,np.mean(newarr))
            if minh < tempo:
                ind = ctr
            ctr +=1
        #print(newarr)
        self.inputarr = temp
        return np.mean(lis),minh,ind


    def test_on_pareto_patch_correctone(self,pareto_set, log_correct = None):
        temp = self.inputarr
        temper = copy.deepcopy(self.testx)
        grand_lis  = []
        for row in temper:
            row_matrix = row.reshape((1, row.shape[0]))
            lis = []
            self.inputarr = row_matrix
            for chromo in pareto_set:

                arr = self.feedforward_ne(chromo)
                assert (arr.shape[0] == 1)
                lis.append(list(arr.reshape((arr.shape[1], ))))
            output_of_all_nn_on_one_data_point = np.array(lis)
            activated_output_of_all_nn_on_one_data_point = softmax(output_of_all_nn_on_one_data_point)
            avrg = np.mean( activated_output_of_all_nn_on_one_data_point, axis = 0)
            argmax_at_avg = avrg.argmax()
            grand_lis.append(argmax_at_avg)
        grand_lis_arr = np.array(grand_lis)
        assert (grand_lis_arr.shape == self.testy.shape)
        if log_correct is not None:
            st = '\n\n'
            for i in range( self.testy.shape[0]):
                if self.testy[i] - grand_lis_arr[i] == 0:
                    print("correct ", self.testy[i])
                    st += "correct " + str(self.testy[i])+'\n'
            st+='\n'
            file_ob = open("./log_folder/log_correct.txt", "a")
            file_ob.write(st)
            file_ob.close()

        difference = self.testy - grand_lis_arr

        to_find_mean_arr = np.where( difference != 0, 1, 0 )
        self.inputarr = temp
        return np.mean(to_find_mean_arr) 








def squa_test(x):
    return (x ** 2).sum(axis=1)

def dummy_popultation(number):#return list of chromosomes
    chromolis=[]
    for i in range(number):
        newchromo=chromosome.Chromosome(0)
        newchromo.rand_init()
        chromolis.append(newchromo)
    return chromolis




