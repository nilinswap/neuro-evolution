# chromosome.py
import math
import random
import gene
from gene import *
import matenc
import numpy as np
import tensorflow as tf
import deep_net
import time

inputnumber = 32
outputnumber = 5  # here could be an error, after all that's why I don't use global variables
innov_ctr = inputnumber * outputnumber + 1


# import network


class Chromosome:
    """
    def __init__(self,dob,node_arr=[],conn_arr=[],bias_arr=[]):
        self.node_arr = node_arr    #list of node objects
        self.conn_arr = conn_arr    #list of conn objects
        self.bias_conn_arr = bias_arr   #list of BiasNode objects
        self.dob = dob              #the generation in which it was created.
        self.node_ctr=len(node_arr)+1
    """

    # here initialization is always with simplest chromosome (AND mainly for innov ctr) , here could be an error
    def __init__(self, inputdim, outputdim, old_chromosome = None):
        if old_chromosome == None:
            global innov_ctr
            self.node_ctr = inputdim + outputdim + 1
            # NO MORE  # Warning!! these two lines change(reset) global variables, here might be some error
            lisI = [gene.Node(num_setter, 'I') for num_setter in range(1, self.node_ctr - outputdim)]
            lisO = [gene.Node(num_setter, 'O') for num_setter in range(inputdim + 1, self.node_ctr)]
            self.node_arr = lisI + lisO
            self.conn_arr = []
            p = 1
            for inputt in lisI:
                for outputt in lisO:
                    self.conn_arr.append(gene.Conn(p, (inputt, outputt), random.random(), status=True))
                    p += 1

                # print(p)
                # assert (p == innov_ctr)
            self.bias_conn_arr = []
            self.bias_conn_arr = [gene.BiasConn(outputt, random.random() / 1000) for outputt in lisO]
            self.dob = 0
        else:
            self.node_ctr = old_chromosome.node_ctr
            self.conn_arr = old_chromosome.conn_arr
            self.bias_conn_arr = old_chromosome.bias_conn_arr
            self.node_arr = old_chromosome.node_arr
            self.dob = old_chromosome.dob



    def reset_chromo_to_zero(self):
        self.node_ctr = 0
        self.node_arr = []
        self.conn_arr = []
        self.bias_conn_arr = []

    def set_node_ctr(self, ctr=None):
        if not ctr:
            ctr = len(self.node_arr) + 1
        self.node_ctr = ctr

    def pp(self):
        print("\nNode List")
        [item.pp() for item in self.node_arr]

        print("\n\nConnection List")
        [item.pp() for item in self.conn_arr]

        print("\n\nBias Connection List")
        [item.pp() for item in self.bias_conn_arr]
        print("dob", self.dob, "node counter", self.node_ctr)
        print("--------------------------------------------")

    def convert_to_MatEnc(self, inputdim, outputdim):

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
        couple_to_conn_map = {}

        for i in self.node_arr:
            dictionary[i.nature][i] = NatureCtrDict[i.nature]
            NatureCtrDict[i.nature] += 1
        """
        ConnMatrix['IO'] = np.zeros((inputdim, outputdim))
        ConnMatrix['IH1'] = np.zeros((inputdim, NatureCtrDict['H1']))
        ConnMatrix['IH2'] = np.zeros((inputdim, NatureCtrDict['H2']))
        ConnMatrix['H1H2'] = np.zeros((NatureCtrDict['H1'], NatureCtrDict['H2']))
        ConnMatrix['H1O'] = np.zeros((NatureCtrDict['H1'], outputdim))
        ConnMatrix['H2O'] = np.zeros((NatureCtrDict['H2'], outputdim))

        WeightMatrix['IO'] = np.zeros((inputdim, outputdim))
        WeightMatrix['IH1'] = np.zeros((inputdim, NatureCtrDict['H1']))
        WeightMatrix['IH2'] = np.zeros((inputdim, NatureCtrDict['H2']))
        WeightMatrix['H1H2'] = np.zeros((NatureCtrDict['H1'], NatureCtrDict['H2']))
        WeightMatrix['H1O'] = np.zeros((NatureCtrDict['H1'], outputdim))
        WeightMatrix['H2O'] = np.zeros((NatureCtrDict['H2'], outputdim))
        """
        for con in self.conn_arr:

            if con.source.nature + con.destination.nature not in ConnMatrix.keys():
                ConnMatrix[con.source.nature + con.destination.nature] = np.zeros(
                    (NatureCtrDict[con.source.nature], NatureCtrDict[con.destination.nature]))
                WeightMatrix[con.source.nature + con.destination.nature] = np.zeros(
                    (NatureCtrDict[con.source.nature], NatureCtrDict[con.destination.nature]))
            if con.status == 1:
                ConnMatrix[con.source.nature + con.destination.nature][
                    dictionary[con.source.nature][con.source]][
                    dictionary[con.destination.nature][con.destination]] = 1
            couple_to_conn_map[con.get_couple()] = con
            # print(con.source.nature + con.destination.nature)
            WeightMatrix[con.source.nature + con.destination.nature][dictionary[con.source.nature][con.source]][
                dictionary[con.destination.nature][con.destination]] = con.weight

        inv_dic = {key: {v: k for k, v in dictionary[key].items()} for key in dictionary.keys()}

        new_encoding = matenc.MatEnc(WeightMatrix, ConnMatrix, self.bias_conn_arr, inv_dic, couple_to_conn_map,
                                     self.node_arr, self.conn_arr)

        return new_encoding

    def modify_thru_backprop(self, inputdim, outputdim, trainx, trainy, epochs=10, learning_rate=0.1, n_par=10):

        x = tf.placeholder(shape=[None, inputdim], dtype=tf.float32)
        y = tf.placeholder(shape=[None, ], dtype=tf.int32)
        n_par = n_par
        par_size = tf.shape(trainx)[0] // n_par
        prmsdind = tf.placeholder(name='prmsdind', dtype=tf.int32)
        valid_x_to_be = trainx[prmsdind * par_size:(prmsdind + 1) * par_size, :]
        valid_y_to_be = trainy[prmsdind * par_size:(prmsdind + 1) * par_size]
        train_x_to_be = tf.concat(
            (trainx[:(prmsdind) * par_size, :], trainx[(prmsdind + 1) * par_size:, :]),
            axis=0)
        train_y_to_be = tf.concat(
            (trainy[:(prmsdind) * par_size], trainy[(prmsdind + 1) * par_size:]), axis=0)

        mat_enc = self.convert_to_MatEnc(inputdim, outputdim)
        newneu_net = deep_net.DeepNet(x, inputdim, outputdim, mat_enc)

        cost = newneu_net.negative_log_likelihood(y)
        # print(newneu_net.mat_enc.CMatrix['IO'])
        optmzr = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=newneu_net.params)
        # savo1 = tf.train.Saver(var_list=[self.srest_setx, self.srest_sety, self.stest_setx, self.stest_sety])
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # err = sess.run(newneu_net.errors(y), feed_dict={x: trainx, y: trainy})
            # print("train error ", err)



            # just any no. which does not satisfy below condition

            prev = 7
            current = 5
            start1 = time.time()
            for epoch in range(epochs):
                listisi = []
                for ind in range(n_par):
                    _, bost = sess.run([optmzr, cost],
                                       feed_dict={x: train_x_to_be.eval(feed_dict={prmsdind: ind}),
                                                  y: train_y_to_be.eval(feed_dict={prmsdind: ind})})

                    if epoch % (epochs // 4) == 0:
                        q = newneu_net.errors(y).eval(
                            feed_dict={x: valid_x_to_be.eval(feed_dict={prmsdind: ind}),
                                       y: valid_y_to_be.eval(feed_dict={prmsdind: ind})})
                        listisi.append(q)
                if epoch % (epochs // 4) == 0:
                    prev = current
                    current = np.mean(listisi)
                    print('validation', current)
                    print(tf.reduce_sum(newneu_net.wei_mat_var_map['IO']).eval())

                if current - prev > 0.0002:
                    break;
            end1 = time.time()
            print("time ", end1 - start1)
            print("now ending")
            for key in newneu_net.wei_mat_var_map.keys():
                newneu_net.mat_enc.WMatrix[key] = newneu_net.wei_mat_var_map[key].eval()
            for i in range(len(newneu_net.bias_wei_arr)):
                ar = newneu_net.bias_var.eval()
                newneu_net.mat_enc.Bias_conn_arr[i].set_weight(ar[i])
        #print(newneu_net.mat_enc.CMatrix['IO'], 'final')
        newchromo = newneu_net.mat_enc.convert_to_chromosome(inputdim, outputdim, self.dob)

        self.conn_arr = newchromo.conn_arr
        self.node_arr = newchromo.node_arr
        self.bias_conn_arr = newchromo.bias_conn_arr  # list of BiasNode objects
        self.dob = newchromo.dob  # the generation in which it was created.
        self.node_ctr = len(self.node_arr) + 1

        return newchromo

    def weight_mutation(self, rng, factor=0.01, individual_change_probablity=0.1):
        import copy
        lis = self.conn_arr + self.bias_conn_arr
        # chosen_ind = rng.choice(range(len(lis)))
        for item in lis:
            if rng.random() < individual_change_probablity:
                item.weight += (rng.random() - 0.5) * 2 * factor
                # return chosen_ind

    def edge_mutation(self, inputdim, outputdim, rng):  # not tested, might just have some error!

        newmatenc = self.convert_to_MatEnc(inputdim, outputdim)
        key_list = list(newmatenc.WMatrix.keys())
        # key_list.remove('IO')
        # print(key_list)


        chosen_key = rng.choice(key_list)
        while chosen_key not in newmatenc.CMatrix.keys():
            chosen_key = rng.choice(key_list)

        mat = newmatenc.CMatrix[chosen_key]
        # print(chosen_key, mat.shape, list(newmatenc.node_map.items()))
        i = rng.randint(0, mat.shape[0]-1)
        if mat.shape[1] > 1:
            j = rng.randint(0, mat.shape[1]-1)
        else:
            j = 0
        split_key1, split_key2 = matenc.split_key(chosen_key)
        # print(split_key1, split_key2)
        # couple = (newmatenc.node_map[split_key1][i], newmatenc.node_map[split_key2][j])
        ctr = 0
        # print(mat[i][j])
        while mat[i][j] != 0:

            i = rng.randint(0, mat.shape[0]-1)
            if mat.shape[1] > 1:
                j = rng.randint(0, mat.shape[1]-1)
            else:
                j = 0
            if ctr > 10:
                return
            ctr += 1
        couple = (newmatenc.node_map[split_key1][i], newmatenc.node_map[split_key2][j])
        mat[i][j] = 1
        if not newmatenc.WMatrix[chosen_key][i][j]:

            innov_num = normalize_conn_arr_for_this_gen(self, couple)
            con_obj = gene.Conn(innov_num, couple, (rng.random() - 0.5) * 2, True)

            self.conn_arr.append(con_obj)

        # con_obj.pp()
        else:

            con_obj = newmatenc.couple_to_conn_map[couple]
            con_obj.status = True

    def node_mutation(self, inputdim, outputdim, rng):
        # global innov_ctr
        type = 0
        newmatenc = self.convert_to_MatEnc(inputdim, outputdim)
        key_list = ['IH2', 'H1O', 'IO']
        stlis = ['H1', 'H2', 'H2']
        # prob_list = [0.1, 0.3, 0.6]
        prndm = rng.random()
        if prndm > 0.4:
            ind = 2
        elif prndm > 0.1:
            ind = 1
        elif prndm > 0:
            ind = 0
        chosen_key = key_list[ind]
        while chosen_key not in newmatenc.CMatrix.keys():
            prndm = rng.random()
            if prndm > 0.4:
                ind = 2
            elif prndm > 0.1:
                ind = 1
            elif prndm > 0:
                ind = 0
            chosen_key = key_list[ind]
        # key_list.remove('IO')
        # print(key_list)

        # chosen_key = (key_list)

        mat = newmatenc.CMatrix[chosen_key]
        # print(chosen_key, mat.shape, list(newmatenc.node_map.items()))
        i = rng.randint(0, mat.shape[0]-1)
        if mat.shape[1] > 1:
            j = rng.randint(0, mat.shape[1]-1)
        else:
            j = 0
        split_key1, split_key2 = matenc.split_key(chosen_key)
        # (split_key1, split_key2)
        ctr = 0

        if not newmatenc.WMatrix[chosen_key][i][j] and not type:
            while mat[i][j] == 0:
                i = rng.randint(0, mat.shape[0]-1)
                if mat.shape[1] > 1:
                    j = rng.randint(0, mat.shape[1]-1)
                else:
                    j = 0
                if ctr > 10:
                    return

                ctr += 1
        couple = (newmatenc.node_map[split_key1][i], newmatenc.node_map[split_key2][j])

        con_obj = newmatenc.couple_to_conn_map[couple]
        con_obj.status = False

        newnode = gene.Node(self.node_ctr, stlis[ind])
        self.node_ctr += 1

        innov_num = normalize_conn_arr_for_this_gen(self, (con_obj.source, newnode))
        new_conn1 = gene.Conn(innov_num, (con_obj.source, newnode), 1.0, True)
        innov_num = normalize_conn_arr_for_this_gen(self, (newnode, con_obj.destination))
        new_conn2 = gene.Conn(innov_num, (newnode, con_obj.destination), con_obj.weight, True)

        self.node_arr.append(newnode)
        self.conn_arr.append(new_conn1)
        self.conn_arr.append(new_conn2)

    def do_mutation(self, rate_conn_weight, rate_conn_itself, rate_node, weight_factor , inputdim, outputdim, max_hidden_unit, rng):
        # rate_conn_weight > rate_conn_itself >> rate_node
        # 0.2, 0.1, 0.05
        p = len(self.conn_arr)
        flag = 0
        rate_conn_itself += rate_node
        rate_conn_weight += rate_conn_itself
        prnd = rng.random()
        if prnd < rate_node:
            if self.node_ctr <= (max_hidden_unit + inputdim + outputdim):
                self.node_mutation(inputdim, outputdim, rng)
                flag = 1
        elif prnd < rate_conn_itself:
            self.edge_mutation(inputdim, outputdim, rng)
            flag = 1
        elif prnd < rate_conn_weight:
            self.weight_mutation(rng, weight_factor)
            flag = 1
        """if flag:
            print("before mutation length", p)
            print("after mutation length", len(self.conn_arr))
        """

    def convert_to_empirical_string(self):

        st = ''

        for con in self.conn_arr:
            tup = con.get_couple()
            st += tup[0].nature + tup[1].nature + str(con.innov_num)
        return st




def normalize_conn_arr_for_this_gen(chromo, tup):
    st = chromo.convert_to_empirical_string()

    global innov_ctr
    if (st, (tup[0].node_num, tup[1].node_num)) in gene.dict_of_sm_so_far.keys():
        innov_num = gene.dict_of_sm_so_far[(st, (tup[0].node_num, tup[1].node_num))]
        # print("matches")
    else:

        innov_num = innov_ctr
        gene.dict_of_sm_so_far[(st, (tup[0].node_num, tup[1].node_num))] = innov_ctr
        innov_ctr += 1
    # print([item for item in gene.dict_of_sm_so_far.items()])
    return innov_num


# 0.5, 0.1, 0.2, 0.2
def aux_weighted(parent1, parent2):
    fitness_tup1 = parent1.fitness
    fitness_tup2 = parent2.fitness
    if fitness_tup1 <= fitness_tup2:
        return parent1
    elif fitness_tup1 > fitness_tup2:
        return parent2
    else:
        theta = 0.5 * fitness_tup1[0] + 0.0001 * fitness_tup1[1] + 0.2 * fitness_tup1[2] + 0.2 * fitness_tup1[3]
        fi = 0.5 * fitness_tup2[0] + 0.0001 * fitness_tup2[1] + 0.2 * fitness_tup2[2] + 0.2 * fitness_tup2[3]
        if theta < fi:
            return parent1
        else:
            return parent2


'''
def aux_weightedTest(parentx, parenty):  # parentx, parenty represents a tuple (parent, fitness_arr)
	parent1 = parentx[0]
	parent2 = parenty[0]
	fitness_tup1 = parentx[1]
	fitness_tup2 = parenty[1]
	if fitness_tup1 <= fitness_tup2:
		return parent1
	elif fitness_tup1 > fitness_tup2:
		return parent2
	else:
		theta = 0.5 * fitness_tup1[0] + 0.0001 * fitness_tup1[1] + 0.2 * fitness_tup1[2] + 0.2 * fitness_tup1[3]
		fi = 0.5 * fitness_tup2[0] + 0.0001 * fitness_tup2[1] + 0.2 * fitness_tup2[2] + 0.2 * fitness_tup2[3]
		if theta < fi:
			return parent1
		else:
			return parent2
'''


def aux_non_weighted(parent1, parent2):
    fitness_tup1 = parent1.fitness
    fitness_tup2 = parent2.fitness

    arr1 = np.array(fitness_tup1)
    arr2 = np.array(fitness_tup2)
    if np.all(arr1 <= arr2):
        return parent1
    elif np.all(arr1 > arr2):
        return parent2
    else:
        if fitness_tup1[0] <= fitness_tup2[0] and np.all(arr1[2:4] <= arr2[2:4]):
            return parent1
        elif fitness_tup1[0] > fitness_tup2[0] and np.all(arr1[2:4] > arr2[2:4]):
            return parent2
        elif ((fitness_tup1[0] <= fitness_tup2[0]) and (
                    (fitness_tup1[2] <= fitness_tup2[2]) or (fitness_tup1[3] <= fitness_tup2[3]))):
            return parent1
        elif ((fitness_tup1[0] > fitness_tup2[0]) and (
                    (fitness_tup1[2] > fitness_tup2[2]) or (fitness_tup1[3] > fitness_tup2[3]))):
            return parent2
        else:
            return random.choice((parent1, parent2))

def aux_non_weighted_1(parent1, parent2):
    fitness_tup1 = parent1.fitness
    fitness_tup2 = parent2.fitness

    arr1 = np.array(fitness_tup1)
    arr2 = np.array(fitness_tup2)
    if np.all(arr1 <= arr2):
        return parent1
    elif np.all(arr1 > arr2):
        return parent2
    else:

        return random.choice((parent1, parent2))
def aux_non_weightedTest(parentx, parenty):
    fitness_tup1 = parentx[1]
    fitness_tup2 = parenty[1]
    parent1 = parentx[0]
    parent2 = parenty[0]
    arr1 = np.array(fitness_tup1)
    arr2 = np.array(fitness_tup2)
    if np.all(arr1 <= arr2):
        return parent1
    elif np.all(arr1 > arr2):
        return parent2
    else:
        if fitness_tup1[0] <= fitness_tup2[0] and np.all(arr1[2:] <= arr2[2:]):
            return parent1
        elif fitness_tup1[0] > fitness_tup2[0] and np.all(arr1[2:] > arr2[2:]):
            return parent2
        elif ((fitness_tup1[0] <= fitness_tup2[0]) and (
                    (fitness_tup1[2] <= fitness_tup2[2]) or (fitness_tup1[3] <= fitness_tup2[3]))):
            return parent1
        elif ((fitness_tup1[0] > fitness_tup2[0]) and (
                    (fitness_tup1[2] > fitness_tup2[2]) or (fitness_tup1[3] > fitness_tup2[3]))):
            return parent2
        else:
            return random.choice((parent1, parent2))


def crossover(parent1, parent2, gen_no, inputdim, outputdim):
    # print("cross between lengths", len(parent1.conn_arr), len(parent2.conn_arr))
    if gen_no > gene.curr_gen_no:
        gene.dict_of_sm_so_far = {}
        gene.curr_gen_no = gen_no
        # print("yes changed",gene.curr_gen_no)

    child = Chromosome(inputdim, outputdim)
    child.reset_chromo_to_zero()
    child.dob = gen_no + 1

    len1 = len(parent1.conn_arr)
    len2 = len(parent2.conn_arr)
    nodeDict = {}
    c1 = 0
    c2 = 0
    dominating_parent = None
    input_nodes = []
    output_nodes = []
    hidden_nodes = []
    while c1 < len1 or c2 < len2:
        f1 = f2 = 0
        if c1 < len1:
            i = parent1.conn_arr[c1]
            f1 = 1
        if c2 < len2:
            j = parent2.conn_arr[c2]
            f2 = 1

        if (f1 == 1 and f1 == f2 and i.innov_num == j.innov_num):
            alpha = random.uniform(0, 1)
            wt = alpha * i.weight + (1 - alpha) * j.weight
            stat = False
            if i.status == j.status:
                stat = i.status
            else:
                stat = random.choice((True, False))

            nodeObj1 = nodeObj2 = None
            if i.source.node_num not in nodeDict.keys():
                nodeObj1 = Node(i.source.node_num, i.source.nature)
                nodeDict[i.source.node_num] = nodeObj1
                if i.source.nature == 'I':
                    input_nodes.append(nodeObj1)
                else:
                    hidden_nodes.append(nodeObj1)

            else:
                nodeObj1 = nodeDict[i.source.node_num]

            if i.destination.node_num not in nodeDict.keys():
                nodeObj2 = Node(i.destination.node_num, i.destination.nature)
                nodeDict[i.destination.node_num] = nodeObj2
                if i.destination.nature == 'H1' or i.destination.nature == 'H2':
                    hidden_nodes.append(nodeObj2)
                else:
                    output_nodes.append(nodeObj2)
            else:
                nodeObj2 = nodeDict[i.destination.node_num]

            conObj = Conn(i.innov_num, (nodeObj1, nodeObj2), wt, stat)  # conn object
            child.conn_arr.append(conObj)
            c1 += 1
            c2 += 1
        else:
            dominating_parent = aux_non_weighted_1(parent1, parent2)
            length = len(dominating_parent.conn_arr)
            while c1 < length:
                i = dominating_parent.conn_arr[c1]
                nodeObj1 = nodeObj2 = None
                if i.source.node_num not in nodeDict.keys():
                    nodeObj1 = Node(i.source.node_num, i.source.nature)
                    nodeDict[i.source.node_num] = nodeObj1
                    if i.source.nature == 'I':
                        input_nodes.append(nodeObj1)
                    else:
                        hidden_nodes.append(nodeObj1)
                else:
                    nodeObj1 = nodeDict[i.source.node_num]

                if i.destination.node_num not in nodeDict.keys():
                    nodeObj2 = Node(i.destination.node_num, i.destination.nature)
                    nodeDict[i.destination.node_num] = nodeObj2
                    if i.destination.nature == 'H1' or i.destination.nature == 'H2':
                        hidden_nodes.append(nodeObj2)
                    else:
                        output_nodes.append(nodeObj2)
                else:
                    nodeObj2 = nodeDict[i.destination.node_num]
                c1 += 1
                connObj = Conn(i.innov_num, (nodeObj1, nodeObj2), i.weight, i.status)
                child.conn_arr.append(connObj)

            break

    input_nodes.sort(key=lambda x: x.node_num)
    output_nodes.sort(key=lambda x: x.node_num)

    child.node_arr = input_nodes + output_nodes + hidden_nodes
    if (outputdim != 1):

        point_of_crossover = random.randint(0, outputdim)

        for i in range(len(output_nodes)):
            if i < point_of_crossover:
                wt = parent1.bias_conn_arr[i].weight
            else:
                wt = parent2.bias_conn_arr[i].weight
            new_bias_conn = gene.BiasConn(output_nodes[i], wt)
            child.bias_conn_arr.append(new_bias_conn)
    elif outputdim == 1:
        p = random.random()
        if p > 0.5:
            wt = parent1.bias_conn_arr[0].weight
        else:
            wt = parent2.bias_conn_arr[0].weight
        new_bias_conn = gene.BiasConn(output_nodes[0], wt)
        child.bias_conn_arr.append(new_bias_conn)

    child.set_node_ctr()
    return child
    """
    assert ( parent1.node_ctr == child.node_ctr or parent2.node_ctr == child.node_ctr)
    assert ( len(parent1.conn_arr) == len(child.conn_arr) or len(parent2.conn_arr) == len(child.conn_arr))
    if dominating_parent:
        #print("FOUND ONE")
        for i in range(len(dominating_parent.conn_arr)):
            assert( dominating_parent.conn_arr[i].innov_num == child.conn_arr[i].innov_num)
            assert( dominating_parent.conn_arr[i].source.nature + dominating_parent.conn_arr[i].destination.nature == child.conn_arr[i].source.nature + child.conn_arr[i].destination.nature)
        assert ( set([item.node_num for item in dominating_parent.node_arr]) == set([item.node_num for item in child.node_arr])  )
    return child
    """


def crossoverTest(parentx, parenty, gen_no, inputdim, outputdim):
    parent1 = parentx[0]
    parent2 = parenty[0]

    # if gen_no > gene.curr_gen_num:
    #   gene.dict_of_sm_so_far = {}
    #   gene.curr_gen_num = gen_no

    child = Chromosome(inputdim, outputdim)
    child.reset_chromo_to_zero()
    child.dob = gen_no

    len1 = len(parent1.conn_arr)
    len2 = len(parent2.conn_arr)
    nodeDict = {}
    c1 = 0
    c2 = 0

    input_nodes = []
    output_nodes = []
    hidden_nodes = []
    while c1 < len1 or c2 < len2:
        f1 = f2 = 0
        if c1 < len1:
            i = parent1.conn_arr[c1]
            f1 = 1
        if c2 < len2:
            j = parent2.conn_arr[c2]
            f2 = 1

        if (f1 == 1 and f1 == f2 and i.innov_num == j.innov_num):
            alpha = random.uniform(0, 1)
            print("alpha", alpha)
            wt = alpha * i.weight + (1 - alpha) * j.weight
            stat = False
            if i.status == j.status:
                stat = i.status
            else:
                stat = random.choice((True, False))
            print(stat)
            nodeObj1 = nodeObj2 = None
            if i.source.node_num not in nodeDict.keys():
                nodeObj1 = Node(i.source.node_num, i.source.nature)
                nodeDict[i.source.node_num] = nodeObj1
                if i.source.nature == 'I':
                    input_nodes.append(nodeObj1)
                else:
                    hidden_nodes.append(nodeObj1)

            else:
                nodeObj1 = nodeDict[i.source.node_num]

            if i.destination.node_num not in nodeDict.keys():
                nodeObj2 = Node(i.destination.node_num, i.destination.nature)
                nodeDict[i.destination.node_num] = nodeObj2
                if i.destination.nature == 'H1' or i.destination.nature == 'H2':
                    hidden_nodes.append(nodeObj2)
                else:
                    output_nodes.append(nodeObj2)
            else:
                nodeObj2 = nodeDict[i.destination.node_num]

            conObj = Conn(i.innov_num, (nodeObj1, nodeObj2), wt, stat)  # conn object
            child.conn_arr.append(conObj)
            c1 += 1
            c2 += 1
        else:
            dominating_parent = aux_non_weightedTest(parentx, parenty)

            length = len(dominating_parent.conn_arr)
            while c1 < length:
                i = dominating_parent.conn_arr[c1]
                nodeObj1 = nodeObj2 = None
                if i.source.node_num not in nodeDict.keys():
                    nodeObj1 = Node(i.source.node_num, i.source.nature)
                    nodeDict[i.source.node_num] = nodeObj1
                    if i.source.nature == 'I':
                        input_nodes.append(nodeObj1)
                    else:
                        hidden_nodes.append(nodeObj1)
                else:
                    nodeObj1 = nodeDict[i.source.node_num]

                if i.destination.node_num not in nodeDict.keys():
                    nodeObj2 = Node(i.destination.node_num, i.destination.nature)
                    nodeDict[i.destination.node_num] = nodeObj2
                    if i.destination.nature == 'H1' or i.destination.nature == 'H2':
                        hidden_nodes.append(nodeObj2)
                    else:
                        output_nodes.append(nodeObj2)
                else:
                    nodeObj2 = nodeDict[i.destination.node_num]
                c1 += 1
                connObj = Conn(i.innov_num, (nodeObj1, nodeObj2), i.weight, i.status)
                child.conn_arr.append(connObj)

            break

    input_nodes.sort(key=lambda x: x.node_num)
    output_nodes.sort(key=lambda x: x.node_num)

    child.node_arr = input_nodes + output_nodes + hidden_nodes

    point_of_crossover = random.randint(1, outputdim - 1)

    for i in range(len(output_nodes)):
        if i < point_of_crossover:
            wt = parent1.bias_conn_arr[i].weight
        else:
            wt = parent2.bias_conn_arr[i].weight
        new_bias_conn = gene.BiasConn(output_nodes[i], wt)
        # Conn(-1, (Node(-1, -1), output_nodes[i]), wt, True)
        child.bias_conn_arr.append(new_bias_conn)

    child.set_node_ctr()
    return child
