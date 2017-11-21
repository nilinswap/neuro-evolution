import numpy as np
# import tf_mlp
import tensorflow as tf
import time
import gene
import matenc
import chromosome
import pimadataf
import deep_net
from chromosome import *
import copy
import population
import network
import cluster

def test1():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    for_node += [(i, 'H2') for i in range(6, 8)]
    node_ctr = 8
    innov_num = 11
    dob = 0
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.4, False), (2, (1, 5), 0.25, True), (3, (2, 4), 0.25, True), (4, (2, 5), 0.5, True),
                (5, (3, 4), 0.7, True),
                (6, (3, 5), 0.6, False), (7, (1, 6), 0.5, True), (8, (6, 4), 0.4, True), (9, (3, 7), 0.3, True),
                (10, (7, 5), 0.6, True)
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
            output1 = sigmoid(relu(arr[0] * 0.5) * 0.4 + 0.25 * arr[1] + 0.7 * arr[2] - 0.2)
            output2 = sigmoid(arr[0] * 0.25 + arr[1] * 0.5 + relu(arr[2] * 0.3) * 0.6 - 0.1)
            lis.append([output1, output2])
        return np.array(lis)

    inputarr = np.array([[3, 2, 1], [4, 1, 2]])
    indim = 3
    outdim = 2
    np.random.seed(4)
    num_data = 2
    # inputarr = np.random.random((num_data, indim))
    neter = Neterr(indim, outdim, inputarr, 10, np.random)
    print(neter.feedforward_ne(newchromo4))
    print(calc_output_directly(inputarr))
    print(neter.feedforward_cm(newchromo4))


# print(neter.feedforward_ne(newchromo, play =1))


def test2():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '2212211'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    indim = 8
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (19, (2, 12), 0.6, True),
                (20, (12, 7), 0.4, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True), (24, (11, 5), 0.25, True),
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo = Chromosome(indim, outdim)
    newchromo.__setattr__('conn_arr', conn_lis)
    newchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo.__setattr__('node_arr', node_lis)
    newchromo.__setattr__('dob', dob)
    newchromo.set_node_ctr(node_ctr)

    def calc_output_directly(inputarr):
        lis = []
        for arr in inputarr:
            x1 = arr[0]
            x2 = arr[1]
            x3 = arr[2]
            output1 = sigmoid(
                0.3 * x1 +
                0.1 * relu(
                    0.7 * relu(0.5 * x1) +
                    0.2 * x1
                ) +
                0.15 * relu(
                    0.1 * x2 +
                    0.4 * relu(
                        0.6 * x2 +
                        0.8 * x3
                    )
                ) +
                0.75 * relu(
                    0.6 * x2 +
                    0.8 * x3
                ) -
                0.2
            )
            # output2 = sigmoid(arr[0] * 0.25 + arr[1] * 0.5 + relu(arr[2] * 0.3) * 0.6 - 0.1)
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

    np.random.seed(4)
    num_data = 2
    # inputarr = np.random.random((num_data, indim))
    neter = Neterr(indim, outdim, 10, np.random)
    print(neter.feedforward_ne(newchromo))
    print(calc_output_directly(inputarr))
    tempchromo = newchromo
    print(neter.feedforward_cm(newchromo))
    if (newchromo == tempchromo):
        print("yeah they are equal")
    print(neter.feedforward_ne(newchromo, play=1))
    print(neter.feedforward_cm(newchromo, play=1))
    print(neter.feedforward_cm(newchromo, play=1))
    print ("done right")

    def interchanging_test(chromo):
        new_mat_enc = chromo.convert_to_MatEnc(indim, outdim)
        newchromo = new_mat_enc.convert_to_chromosome(indim, outdim, dob)
        if newchromo.bias_conn_arr != chromo.bias_conn_arr:
            print("falied 1")
        if newchromo.node_arr != chromo.node_arr:
            print("failed 2")
        if newchromo.dob != chromo.dob or newchromo.node_ctr != chromo.node_ctr:
            print("failed 3", "node_ctr are", newchromo.node_ctr, chromo.node_ctr, len(newchromo.node_arr),
                  len(chromo.node_arr))
        listup1 = [(con.status, con.weight, con.get_couple(), con.innov_num) for con in newchromo.conn_arr]
        listup2 = [(con.status, con.weight, con.get_couple(), con.innov_num) for con in chromo.conn_arr]
        # print( set(listup1),set(listup2))
        if set(listup1) != set(listup2):
            print("failed 4")

    interchanging_test(newchromo)


def test_for_muta():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '2212211'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    indim = 8
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (19, (2, 12), 0.6, True),
                (20, (12, 7), 0.4, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True), (24, (11, 5), 0.25, True),
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo = Chromosome(indim, outdim)
    newchromo.__setattr__('conn_arr', conn_lis)
    newchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo.__setattr__('node_arr', node_lis)
    newchromo.__setattr__('dob', dob)
    newchromo.set_node_ctr(node_ctr)
    # chromosome.Chromosome.pp(newchromo)
    import copy
    random_val = 0.9
    zarr = copy.deepcopy(newchromo.conn_arr)
    """if random_val <= 0.8:
        newchromo.weight_mutation(np.random)
    else:
        newchromo.edge_mutation(indim, outdim, np.random)
        pass
    """
    newchromo.do_mutation(1, 1, 1, indim, outdim, np.random)
    # print(zarr[zind].pp(), newchromo.conn_arr[zind].pp())
    print(len(zarr), len(newchromo.conn_arr))
    print("Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    # chromosome.Chromosome.pp(newchromo)
    # newchromo.pp()
    def calc_output_directly(inputarr):
        lis = []
        for arr in inputarr:
            x1 = arr[0]
            x2 = arr[1]
            x3 = arr[2]
            output1 = sigmoid(
                0.3 * x1 +
                0.1 * relu(
                    0.7 * relu(0.5 * x1) +
                    0.2 * x1
                ) +
                0.15 * relu(
                    0.1 * x2 +
                    0.4 * relu(
                        0.6 * x2 +
                        0.8 * x3
                    )
                ) +
                0.75 * relu(
                    0.6 * x2 +
                    0.8 * x3
                ) -
                0.2
            )
            # output2 = sigmoid(arr[0] * 0.25 + arr[1] * 0.5 + relu(arr[2] * 0.3) * 0.6 - 0.1)
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

    np.random.seed(4)
    num_data = 2
    # inputarr = np.random.random((num_data, indim))
    neter = Neterr(indim, outdim, 10, np.random)
    print(neter.feedforward_ne(newchromo))
    print(calc_output_directly(inputarr))
    tempchromo = newchromo
    print(neter.feedforward_cm(newchromo))
    if (newchromo == tempchromo):
        print("yeah they are equal")
    print(neter.feedforward_ne(newchromo, play=1))
    print(neter.feedforward_cm(newchromo, play=1))
    print(neter.feedforward_cm(newchromo, play=1))
    print ("done right")

    def interchanging_test(chromo):
        new_mat_enc = chromo.convert_to_MatEnc(indim, outdim)
        newchromo = new_mat_enc.convert_to_chromosome(indim, outdim, dob)
        if newchromo.bias_conn_arr != chromo.bias_conn_arr:
            print("falied 1")
        if newchromo.node_arr != chromo.node_arr:
            print("failed 2")
        if newchromo.dob != chromo.dob or newchromo.node_ctr != chromo.node_ctr:
            print("failed 3", "node_ctr are", newchromo.node_ctr, chromo.node_ctr, len(newchromo.node_arr),
                  len(chromo.node_arr))
        listup1 = [(con.status, con.weight, con.get_couple(), con.innov_num) for con in newchromo.conn_arr]
        listup2 = [(con.status, con.weight, con.get_couple(), con.innov_num) for con in chromo.conn_arr]
        # print( set(listup1),set(listup2))
        if set(listup1) != set(listup2):
            print("failed 4")

    interchanging_test(newchromo)


def test_mtbp():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '2212211'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True),
                (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (19, (2, 12), 0.6, True),
                (20, (12, 7), 0.4, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True),
                (24, (11, 5), 0.25, True),
        ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in
                for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    indim = 8
    outdim = 2
    newchromo = Chromosome(indim, outdim)
    newchromo.reset_chromo_to_zero()
    newchromo.__setattr__('conn_arr', conn_lis)
    newchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo.__setattr__('node_arr', node_lis)
    newchromo.__setattr__('dob', dob)
    newchromo.set_node_ctr()

    # newchromo.pp()
    def calc_output_directly(inputarr):
        lis = []
        for arr in inputarr:
            x1 = arr[0]
            x2 = arr[1]
            x3 = arr[2]
            output1 = sigmoid(
                0.3 * x1 +
                0.1 * relu(
                    0.7 * relu(0.5 * x1) +
                    0.2 * x1
                ) +
                0.15 * relu(
                    0.1 * x2 +
                    0.4 * relu(
                        0.6 * x2 +
                        0.8 * x3
                    )
                ) +
                0.75 * relu(
                    0.6 * x2 +
                    0.8 * x3
                ) -
                0.2
            )
            # output2 = sigmoid(arr[0] * 0.25 + arr[1] * 0.5 + relu(arr[2] * 0.3) * 0.6 - 0.1)
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

    #inputarr = np.array([[0.0, 2, 1], [0.8, 1, 2]])
    indim = 8
    outdim = 2

    # np.random
    # rng = np.random
    # num_data = 10
    # inputarr = np.random.random((num_data, indim))
    neter = network.Neterr(indim, outdim, 10, np.random)

    # ka = np.random.randint(0, 2, (num_data,))
    # print(neter.feedforward_ne(chromo))
    """
    targetarr = np.zeros((num_data,outdim)).astype(dtype = 'float32')
    for i in range(num_data):
        targetarr[i,ka[i]] = 1

    print("target is ", targetarr)
    """

    #targetarr = ka.astype('int32')
    #print(targetarr.dtype)
    #inputarr = inputarr.astype('float32')

    tempchromo = copy.deepcopy(newchromo)
    arr = newchromo.node_arr
    newmatenc = tempchromo.convert_to_MatEnc(indim, outdim)
    newmatenc = copy.deepcopy(newmatenc)

    print(newmatenc.CMatrix['IO'])

    newchromo.modify_thru_backprop(indim, outdim, neter.rest_setx, neter.rest_sety)
    if not newchromo.node_arr == arr:
        print("failed 1")
    if not newchromo.dob == tempchromo.dob and not newchromo.node_ctr == tempchromo.node_ctr:
        print("failed 2")
    if not len(newchromo.conn_arr) == len(tempchromo.conn_arr):
        print("failed 3", len(newchromo.conn_arr), len(tempchromo.conn_arr))

    newnewmatenc = newchromo.convert_to_MatEnc(indim, outdim)

    for key in newnewmatenc.WMatrix.keys():
        if (newnewmatenc.WMatrix[key] == newmatenc.WMatrix[key]).all():
            print("failed 5", key)


def newtest():
    indim = 8
    outdim = 1

    # np.random
    rng = np.random
    num_data = 10
    # inputarr = np.random.random((num_data, indim))
    neter = network.Neterr(indim, outdim, 10, np.random)

    # ka = np.random.randint(0, 2, (num_data,))
    # print(neter.feedforward_ne(chromo))
    """
    targetarr = np.zeros((num_data,outdim)).astype(dtype = 'float32')
    for i in range(num_data):
        targetarr[i,ka[i]] = 1

    print("target is ", targetarr)
    """

    """
    targetarr = ka.astype('int32')
    print(targetarr.dtype)
    inputarr = inputarr.astype('float32')

    tempchromo = copy.deepcopy(newchromo)
    arr = newchromo.node_arr
    newmatenc = tempchromo.convert_to_MatEnc(indim, outdim)
    newmatenc=copy.deepcopy(newmatenc)

    newchromo.modify_thru_backprop(  indim, outdim, neter.rest_setx, neter.rest_sety)
    if not newchromo.node_arr == arr:
        print("failed 1")
    if not newchromo.dob == tempchromo.dob and not newchromo.node_ctr == tempchromo.node_ctr:
        print("failed 2")
    if not len(newchromo.conn_arr) == len(tempchromo.conn_arr):
        print("failed 3")

    newnewmatenc = newchromo.convert_to_MatEnc(indim,outdim)



    for key in newnewmatenc.WMatrix.keys():
        if (newnewmatenc.WMatrix[key] == newmatenc.WMatrix[key]).all():

            print("failed 5",key)
    """

    popul = population.Population(indim, outdim, 10, 40)
    # popul.set_initial_population_as_list(indim,1,dob=0)
    # [item.pp() for item in popul.list_chromo]
    print(len(popul.list_chromo))
    popul.list_chromo[0].do_mutation(1, 1, 1, 8, 1, np.random)
    # print(popul.list_chromo[0].pp())
    print("-----------------------------------------------------------")
    time.sleep(5)
    # print(popul.list_chromo[2].pp())
    popul.set_objective_arr(neter)
    print(popul.objective_arr)


def test3():
    for_node = [(i, 'I') for i in range(1, 9)]
    for_node += [(i, 'O') for i in range(9, 10)]
    # for_node += [(i, 'H2') for i in range(6, 8)]
    node_ctr = 10
    innov_num = 9
    dob = 0
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(inn, (node_lis[inn - 1], node_lis[8]), np.random.random(), True) for inn in range(1, innov_num)]
    conn_lis = [gene.Conn(*p) for p in for_conn]

    for_bias = [(4, 0.2)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]

    indim = 8
    outdim = 1
    newchromo = Chromosome(indim, outdim)
    newchromo.reset_chromo_to_zero()
    newchromo.__setattr__('conn_arr', conn_lis)
    newchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo.__setattr__('node_arr', node_lis)
    newchromo.__setattr__('dob', dob)
    newchromo.set_node_ctr(node_ctr)

    # newchromo.pp()
    def calc_output_directly(inputarr):
        lis = []
        for arr in inputarr:
            output1 = sigmoid(relu(arr[0] * 0.5) * 0.4 + 0.25 * arr[1] + 0.7 * arr[2] - 0.2)
            output2 = sigmoid(arr[0] * 0.25 + arr[1] * 0.5 + relu(arr[2] * 0.3) * 0.6 - 0.1)
            lis.append([output1, output2])
        return np.array(lis)

    indim = 8
    outdim = 1

    # num_data = 2
    # inputarr = np.random.random((num_data, indim))
    neter = Neterr(indim, outdim, 10, np.random)
    print(neter.feedforward_ne(newchromo))
    # print(calc_output_directly(inputarr))
    print("break")
    print(neter.feedforward_cm(newchromo))


# print(neter.feedforward_ne(newchromo, play =1))
def test_for_cros():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '2212211'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, False), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (19, (2, 12), 0.6, True),
                (20, (12, 7), 0.4, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True), (24, (11, 5), 0.25, True),
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo = Chromosome(indim, outdim)
    newchromo.__setattr__('conn_arr', conn_lis)
    newchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo.__setattr__('node_arr', node_lis)
    newchromo.__setattr__('dob', dob)
    newchromo.set_node_ctr(node_ctr)
    inputarr = np.array([[0, 2, 1], [0.8, 1, 2]])

    np.random.seed(4)
    num_data = 2
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '22122'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 11
    innov_num = 17
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.8, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True)
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newnewchromo = Chromosome(indim, outdim)
    newnewchromo.__setattr__('conn_arr', conn_lis)
    newnewchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newnewchromo.__setattr__('node_arr', node_lis)
    newnewchromo.__setattr__('dob', dob)
    newnewchromo.set_node_ctr(node_ctr)

    # print("newchromo")
    # newchromo.pp()

    # print("newnewchromo")
    # newnewchromo.pp()
    print("final")
    gen_no = 1
    tup2 = (0.4, 0.5, 0.7, 0.8)
    tup1 = (0.6, 0.4, 0.6, 0.7)

    # chromosome.crossover(newchromo, newnewchromo, gen_no).pp()
    chromosome.crossoverTest((newchromo, tup1), (newnewchromo, tup2), gen_no).pp()


def test_for_cros2():

    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '221221'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 12
    innov_num = 21
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (21, (2, 9), 0.65, True)]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo = Chromosome(indim, outdim)
    newchromo.__setattr__('conn_arr', conn_lis)
    newchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo.__setattr__('node_arr', node_lis)
    newchromo.__setattr__('dob', dob)
    newchromo.set_node_ctr(node_ctr)
    inputarr = np.array([[0, 2, 1], [0.8, 1, 2]])

    np.random.seed(4)
    num_data = 2
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '221221'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 12
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (19, (2, 11), 0.6, True), (20, (11, 7), 0.4, True), (22, (3, 11), 0.75, True)]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newnewchromo = Chromosome(indim, outdim)
    newnewchromo.__setattr__('conn_arr', conn_lis)
    newnewchromo.__setattr__('bias_conn_arr', bias_conn_lis)
    newnewchromo.__setattr__('node_arr', node_lis)
    newnewchromo.__setattr__('dob', dob)
    newnewchromo.set_node_ctr(node_ctr)

    # print("newchromo")
    # newchromo.pp()

    # print("newnewchromo")
    # newnewchromo.pp()
    print("final")
    gen_no = 1
    tup1 = (0.4, 0.5, 0.7, 0.8)
    tup2 = (0.6, 0.4, 0.6, 0.7)

    # chromosome.crossover(newchromo, newnewchromo, gen_no).pp()
    # chromosome.crossoverTest((newchromo, tup1), (newnewchromo, tup2), gen_no).pp()
    wt_muta = 0.3
    node_muta = 0.1
    conn_muta = 0.2
    chromosome.do_mutation(wt_muta, conn_muta ,node_muta, indim, outdim, 5)


def test_for_cluster():
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    for_node += [(i, 'H2') for i in range(6, 8)]
    node_ctr = 8
    indim = 3
    outdim =2
    innov_num = 11
    dob = 0
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.4, False), (2, (1, 5), 0.25, True), (3, (2, 4), 0.25, True), (4, (2, 5), 0.5, True),
                (5, (3, 4), 0.7, True),
                (6, (3, 5), 0.6, False), (7, (1, 6), 0.5, True), (8, (6, 4), 0.4, True), (9, (3, 7), 0.3, True),
                (10, (7, 5), 0.6, True)
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]

    newchromo1 = Chromosome(indim, outdim)
    newchromo1.__setattr__('conn_arr', conn_lis)
    newchromo1.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo1.__setattr__('node_arr', node_lis)
    newchromo1.__setattr__('dob', dob)
    newchromo1.set_node_ctr(node_ctr)

    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '2212211'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    indim = 8
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (19, (2, 12), 0.6, True),
                (20, (12, 7), 0.4, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True), (24, (11, 5), 0.25, True),
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo2 = Chromosome(indim, outdim)
    newchromo2.__setattr__('conn_arr', conn_lis)
    newchromo2.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo2.__setattr__('node_arr', node_lis)
    newchromo2.__setattr__('dob', dob)
    newchromo2.set_node_ctr(node_ctr)

    # newchromo.pp()
    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '221221'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 12
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (19, (2, 11), 0.6, True), (20, (11, 7), 0.4, True), (22, (3, 11), 0.75, True)]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo3 = Chromosome(indim, outdim)
    newchromo3.__setattr__('conn_arr', conn_lis)
    newchromo3.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo3.__setattr__('node_arr', node_lis)
    newchromo3.__setattr__('dob', dob)
    newchromo3.set_node_ctr(node_ctr)

    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '221221'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 12
    innov_num = 21
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True), (18, (11, 9), 0.15, True), (21, (2, 9), 0.65, True)]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo4 = Chromosome(indim, outdim)
    newchromo4.__setattr__('conn_arr', conn_lis)
    newchromo4.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo4.__setattr__('node_arr', node_lis)
    newchromo4.__setattr__('dob', dob)
    newchromo4.set_node_ctr(node_ctr)

    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '2212211'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 13
    innov_num = 25
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.3, False), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True),
                (17, (1, 11), 0.25, True),
                (21, (3, 12), 0.8, True), (22, (2, 9), 0.9, True), (23, (12, 4), 0.75, True), (24, (11, 5), 0.25, True),
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo5 = Chromosome(indim, outdim)
    newchromo5.__setattr__('conn_arr', conn_lis)
    newchromo5.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo5.__setattr__('node_arr', node_lis)
    newchromo5.__setattr__('dob', dob)
    newchromo5.set_node_ctr(node_ctr)

    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    st = '22122'
    for_node += [(i + 6, 'H' + st[i]) for i in range(len(st))]
    node_ctr = 11
    innov_num = 17
    dob = 0
    indim = 3
    outdim = 2
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.8, True), (2, (1, 5), 0.25, False), (3, (2, 4), 0.25, False), (4, (2, 5), 0.5, False),
                (5, (3, 4), 0.7, False), (6, (3, 5), 0.5, True), (7, (1, 6), 0.2, True), (8, (6, 4), 0.1, True),
                (9, (2, 7), 0.1, True), (10, (7, 4), 0.15, True), (11, (1, 8), 0.5, True), (12, (8, 6), 0.7, True),
                (13, (1, 9), 0.3, False), (14, (9, 5), 1.0, True), (15, (3, 10), 0.33, True), (16, (10, 5), 0.77, True)
                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo6 = Chromosome(indim, outdim)
    newchromo6.__setattr__('conn_arr', conn_lis)
    newchromo6.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo6.__setattr__('node_arr', node_lis)
    newchromo6.__setattr__('dob', dob)
    newchromo6.set_node_ctr(node_ctr)

    for_node = [(i, 'I') for i in range(1, 4)]
    for_node += [(i, 'O') for i in range(4, 6)]
    for_node += [(i, 'H2') for i in range(6, 8)]
    node_ctr = 8
    indim = 3
    outdim = 2
    innov_num = 11
    dob = 0
    node_lis = [gene.Node(x, y) for x, y in for_node]
    for_conn = [(1, (1, 4), 0.4, False), (2, (1, 5), 0.25, True), (3, (2, 4), 0.25, True), (4, (2, 5), 0.5, True),
                (5, (3, 4), 0.7, True),
                (6, (3, 5), 0.6, False), (7, (1, 6), 0.5, True), (8, (6, 4), 0.4, True), (9, (3, 7), 0.3, True)

                ]
    conn_lis = [gene.Conn(x, (node_lis[tup[0] - 1], node_lis[tup[1] - 1]), w, status) for x, tup, w, status in for_conn]
    for_bias = [(4, 0.2), (5, 0.1)]
    bias_conn_lis = [gene.BiasConn(node_lis[x - 1], y) for x, y in for_bias]
    newchromo7 = Chromosome(indim, outdim)
    newchromo7.__setattr__('conn_arr', conn_lis)
    newchromo7.__setattr__('bias_conn_arr', bias_conn_lis)
    newchromo7.__setattr__('node_arr', node_lis)
    newchromo7.__setattr__('dob', dob)
    newchromo7.set_node_ctr(node_ctr)


    chromo_list = [newchromo1, newchromo2, newchromo3, newchromo4, newchromo5, newchromo6, newchromo7]
    #print([item.pp() for item in chromo_list])

    print( cluster.distance(newchromo3,newchromo2) )

    cluster.give_cluster_head(chromo_list, 2)
    inputarr = np.array([[0, 2, 1], [0.8, 1, 2]])



if __name__ == '__main__':
    test_for_cluster()
