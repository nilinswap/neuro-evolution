from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import time
import numpy as np
import chromosome
import gene
import matenc
import tensorflow as tf

def func(last, current):
    return [last[0] + 1, current]

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

def relu(arr):
    return np.where(arr>0,arr,0)

def dense_to_sparse(mat):
    idx = tf.where(tf.not_equal(mat, 0))
    sparse = tf.SparseTensor(idx, tf.gather_nd(mat, idx), mat.get_shape())
    return sparse

def find_density(mat):
    idx = tf.where(tf.not_equal(mat, 0))
    #val = tf.gather_nd(mat, idx)
    tup = tf.shape(mat)
    return tf.divide( tf.shape(idx)[0],tf.multiply(tup[0],tup[1]))

def opt_compwise_multiply(mat1,mat2):
    tempnode = dense_to_sparse( mat1 ).__mul__( mat2 )
    return tf.scatter_nd(tempnode.indices, tempnode.values, tempnode.dense_shape)



class DeepNet(object):
    def __init__(self, inputh, n_in, n_out, mat_enc, middle_activation = tf.nn.relu, final_activation = tf.nn.sigmoid):


        self.n_in = n_in
        self.n_out = n_out
        #self.chromo = chromo
        self.mat_enc = mat_enc

        self.input = inputh

        self.con_mat_var_map=  {}
        self.wei_mat_var_map = {}
        for key in self.mat_enc.CMatrix.keys():
            self.con_mat_var_map[key] = tf.Variable(initial_value= self.mat_enc.CMatrix[key].astype('float32'), name = 'con_mat'+key, dtype = tf.float32)
            self.wei_mat_var_map[key] = tf.Variable( initial_value = self.mat_enc.WMatrix[key], name = 'con_mat'+key, dtype = tf.float32)

        to_effec_mat_node_map = {}
        for key in  self.con_mat_var_map.keys():
            to_effec_mat_node_map[key] = opt_compwise_multiply(self.con_mat_var_map[ key ],self.wei_mat_var_map[ key ])

        density_map = {}
        for key in to_effec_mat_node_map.keys():
            density_map[key] = find_density(to_effec_mat_node_map[key])
        self.bias_wei_arr = np.array( [ item.weight for item in self.mat_enc.Bias_conn_arr] )
        self.bias_var = tf.Variable( initial_value = self.bias_wei_arr, name  = "bias", dtype = tf.float32)

        input_till_H2 = None

        if 'IH1' in to_effec_mat_node_map.keys():
            input_till_H1 = middle_activation(tf.sparse_matmul( self.input, to_effec_mat_node_map['IH1'], b_is_sparse = True))

        if 'IH2' in to_effec_mat_node_map.keys():
            input_till_H2 =  tf.sparse_matmul(self.input, to_effec_mat_node_map['IH2'], b_is_sparse = True)

        if 'H1H2' in to_effec_mat_node_map.keys():
            assert( 'IH1' in to_effec_mat_node_map.keys())
            twoh = tf.sparse_matmul( input_till_H1, to_effec_mat_node_map['H1H2'], b_is_sparse = True)
            if 'IH2' in to_effec_mat_node_map.keys():
                input_till_H2 =     tf.add(twoh, input_till_H2)

            else:
                input_till_H2 =  twoh

        if input_till_H2 is not None:
            input_till_H2 = middle_activation(input_till_H2)

        output = None
        if 'H2O' in to_effec_mat_node_map.keys():
            assert('IH2' in to_effec_mat_node_map.keys() or 'H1H2' in to_effec_mat_node_map.keys())
            threeh = tf.sparse_matmul(input_till_H2, to_effec_mat_node_map['H2O'], b_is_sparse=True)

            output = threeh

        if 'H1O' in to_effec_mat_node_map.keys():
            assert ('IH1' in to_effec_mat_node_map.keys())
            fourh = tf.sparse_matmul(input_till_H1, to_effec_mat_node_map['H1O'], b_is_sparse = True)

            if output is not None:
                output = tf.add( output, fourh)
            else:
                output = fourh

        if 'IO' in to_effec_mat_node_map.keys():
            assert ('IO' in to_effec_mat_node_map.keys())
            fifth = tf.sparse_matmul(self.input, to_effec_mat_node_map['IO'] , b_is_sparse = True)
            if output is not None:
                output = tf.add( output, fifth)
            else:
                output = fifth

        output = final_activation(output)
        """input_till_H2 = middle_activation(
                            tf.add(
                                tf.sparse_matmul(self.input, to_effec_mat_node_map['IH2'], b_is_sparse = True),
                                tf.sparse_matmul( input_till_H1, to_effec_mat_node_map['H1H2'], b_is_sparse = True)
                            )
                        )


        output    = final_activation(
                        tf.add(
                            tf.add(
                                    tf.add(
                                            tf.sparse_matmul(input_till_H2, to_effec_mat_node_map['H2O'], b_is_sparse = True ),
                                            tf.sparse_matmul(input_till_H1,to_effec_mat_node_map['H1O'], b_is_sparse = True)
                                    ),
                                    tf.sparse_matmul(self.input, to_effec_mat_node_map['IO'] , b_is_sparse = True)
                            ),

                            self.bias_var
                        )
                    )
        """
        self.p_y_given_x = output

        half = tf.constant(0.5, dtype=self.p_y_given_x.dtype)
        if int(self.bias_wei_arr.shape[0]) != 1:
            self.y_pred = tf.argmax(self.p_y_given_x, axis=1)
        else:
            half = tf.constant(0.5, dtype=self.p_y_given_x.dtype)
            dadum = tf.constant(0.5, dtype=self.p_y_given_x.dtype)
            q = tf.scan(lambda last, current: current[0], elems=self.p_y_given_x, initializer=dadum)
            s = tf.scan(lambda y, x: tf.greater_equal(x, half), elems=q, initializer=False)
            #print("herehrerhehrehrehrehrhe", s)
            # print("hi",s)
            self.y_pred = tf.cast(s, dtype=tf.int32)

        self.params = [ self.wei_mat_var_map[key] for key in self.wei_mat_var_map.keys()] + [self.bias_var]


    def negative_log_likelihood(self, y):
        if int(self.bias_wei_arr.shape[0])!=1:
           dum=tf.constant(0.5,dtype=tf.float32) #dum for dummy
           dadum=tf.constant(-1,dtype=tf.int32)# dum-dadum-dadum mast h
           q=tf.scan(fn=func,elems=y,initializer=[dadum,dadum])
           z=tf.transpose(tf.stack([q[0],q[1]]))
           #print("hello---------------------------")
           w=tf.scan(lambda last,current: tf.log(self.p_y_given_x[current[0]][current[1]]),elems=z, initializer = dum)
           #print(-tf.reduce_mean(w))
           return -tf.reduce_mean(w)
        else:

            dum=tf.constant(0.5,dtype=tf.float64)
            minusone=tf.constant(-1,dtype=tf.int32)
            one=tf.constant(1,dtype=y.dtype)
            r=tf.scan(lambda last,current:last+1,elems=y,initializer=minusone)

            w=tf.scan(lambda last,current: tf.add(tf.multiply(tf.cast(y[current],dtype=self.p_y_given_x.dtype),tf.log(self.p_y_given_x[current][0])),tf.multiply(tf.cast(tf.add(one,-y[current]),dtype=self.p_y_given_x.dtype),tf.log(tf.add(tf.cast(one,dtype=self.p_y_given_x.dtype),-self.p_y_given_x[current][0])))),elems=r,initializer=dum)
            z=-tf.reduce_mean(w)
            return z

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        """if len(y.shape) != len(self.y_pred.shape):
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        """
        # check if y is of the correct datatype

        if y.dtype:
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            r = tf.scan(lambda last, current: last + 1, elems=y, initializer=-1)
            qn = tf.scan(lambda last, current: tf.not_equal(tf.cast(self.y_pred[current], dtype=tf.int32), y[current]),
                         elems=r, initializer=False)
            q = tf.cast(qn, dtype=tf.int32)

            # r=tf.scan((lambda last,current: current[1]),q)
            return tf.reduce_mean(tf.cast(q, dtype=tf.float64))
        else:
            raise NotImplementedError()


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
    newchromo = chromosome.Chromosome(dob, node_lis, conn_lis, bias_conn_lis)
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

    inputarr = np.array([[0.0, 2, 1], [0.8, 1, 2]])
    indim = 3
    outdim = 2

    #np.random
    rng = np.random
    num_data = 2
    # inputarr = np.random.random((num_data, indim))
    #neter = Neterr(indim, outdim, inputarr, 10, np.random)

    ka = np.random.randint(0,2,(num_data,))
    """
    targetarr = np.zeros((num_data,outdim)).astype(dtype = 'float32')
    for i in range(num_data):
        targetarr[i,ka[i]] = 1

    print("target is ", targetarr)
    """
    targetarr = ka.astype('int32')
    print(targetarr.dtype)
    inputarr = inputarr.astype('float32')
    print("input type", inputarr.dtype)
    print(targetarr)
    x = tf.placeholder( shape = [None, indim], dtype = tf.float32)
    y = tf.placeholder( shape = [None,], dtype = tf.int32)
    newmat_enc = newchromo.convert_to_MatEnc(indim,outdim)
    newnet = DeepNet(x, indim, outdim, newmat_enc)
    cost = newnet.negative_log_likelihood(y)
    learning_rate = 0.05
    optmzr = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=newnet.params)
    #cost = newnet.errors(y)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print("initially")
        print(sess.run(newnet.wei_mat_var_map['IH2']))
        print(sess.run(newnet.con_mat_var_map['IH2']))
        print(sess.run([optmzr,newnet.y_pred,cost], feed_dict = { x : inputarr, y : targetarr}))
        print(sess.run(newnet.wei_mat_var_map['IH2']))
        print(sess.run([optmzr, newnet.bias_var, cost], feed_dict={x: inputarr, y: targetarr}))
        print(sess.run(newnet.con_mat_var_map['IH2']))
        print(sess.run([optmzr, newnet.bias_var, newnet.errors(y)], feed_dict={x: inputarr, y: targetarr}))


    #newchromo.modify_thru_backprop()

if __name__ == '__main__':
    test2()








