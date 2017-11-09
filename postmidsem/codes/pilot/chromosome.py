# chromosome.py

import gene
import matenc
import numpy as np
import tensorflow as tf
import deep_net
import time

innov_ctr = None




class Chromosome:
    """
    def __init__(self,dob,node_arr=[],conn_arr=[],bias_arr=[]):
        self.node_arr = node_arr	#list of node objects
        self.conn_arr = conn_arr	#list of conn objects
        self.bias_conn_arr = bias_arr	#list of BiasNode objects
        self.dob = dob 				#the generation in which it was created.
        self.node_ctr=len(node_arr)+1
    """
        #here initialization is always with simplest chromosome (AND mainly for innov ctr) , here could be an error
    def __init__(self,inputdim,outputdim):
        global innov_ctr

        self.node_ctr = inputdim + outputdim + 1
        innov_ctr = 1  # Warning!! these two lines change(reset) global variables, here might be some error
        lisI = [gene.Node(num_setter, 'I') for num_setter in range(1, self.node_ctr - outputdim)]
        lisO = [gene.Node(num_setter, 'O') for num_setter in range(inputdim + 1, self.node_ctr)]
        self.node_arr = lisI + lisO
        self.conn_arr=[]
        for inputt in lisI:
            for outputt in lisO:
                self.conn_arr.append(gene.Conn(innov_ctr, (inputt, outputt), np.random.random(), status=True))

                innov_ctr += 1
        self.bias_conn_arr = []
        self.bias_conn_arr = [gene.BiasConn(outputt, np.random.random()/1000) for outputt in lisO]
        self.dob = 0

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
        couple_to_innov_map = {}

        for i in self.node_arr:
            dictionary[i.nature][i] = NatureCtrDict[i.nature]
            NatureCtrDict[i.nature] += 1

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

        for con in self.conn_arr:
            if con.status == True:
                ConnMatrix[con.source.nature + con.destination.nature][
                    dictionary[con.source.nature][con.source]][
                    dictionary[con.destination.nature][con.destination]] = 1
            couple_to_innov_map[con.get_couple()] = con.innov_num
            WeightMatrix[con.source.nature + con.destination.nature][dictionary[con.source.nature][con.source]][
                dictionary[con.destination.nature][con.destination]] = con.weight

        inv_dic = {key: {v: k for k, v in dictionary[key].items()} for key in dictionary.keys()}

        new_encoding = matenc.MatEnc(WeightMatrix, ConnMatrix, self.bias_conn_arr, inv_dic, couple_to_innov_map,
                                     self.node_arr)

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

                if prev - current < 0.002:
                    break;
            end1 = time.time()
            print("time ", end1 - start1)

            for key in newneu_net.wei_mat_var_map.keys():
                newneu_net.mat_enc.WMatrix[key] = newneu_net.wei_mat_var_map[key].eval()
            for i in range(len(newneu_net.bias_wei_arr)):
                ar = newneu_net.bias_var.eval()
                newneu_net.mat_enc.Bias_conn_arr[i].set_weight(ar[i])
        newchromo = newneu_net.mat_enc.convert_to_chromosome(self.dob)

        self.conn_arr = newchromo.conn_arr
        self.node_arr = newchromo.node_arr
        self.bias_conn_arr = newchromo.bias_conn_arr  # list of BiasNode objects
        self.dob = newchromo.dob  # the generation in which it was created.
        self.node_ctr = len(self.node_arr) + 1

        return newchromo

# def rand_init(inputdim, outputdim):
#     global innov_ctr
#     newchromo = Chromosome(0)
#
#     newchromo.node_ctr = inputdim + outputdim + 1
#     innov_ctr = 1  # Warning!! these two lines change(reset) global variables, here might be some error
#     lisI = [gene.Node(num_setter, 'I') for num_setter in range(1, newchromo.node_ctr - outputdim)]
#     lisO = [gene.Node(num_setter, 'O') for num_setter in range(inputdim + 1, newchromo.node_ctr)]
#     newchromo.node_arr = lisI + lisO
#     for inputt in lisI:
#         for outputt in lisO:
#             newchromo.conn_arr.append(gene.Conn(innov_ctr, (inputt, outputt), np.random.random(), status=True))
#             innov_ctr += 1
#     newchromo.bias_arr = [gene.BiasConn(outputt, np.random.random()) for outputt in lisO]
#     newchromo.dob = 0
#     return newchromo
