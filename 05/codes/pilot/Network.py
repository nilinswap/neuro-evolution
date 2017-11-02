import numpy as np
import tf_mlp
import tensorflow as tf
import time
import gene 
import chromosome

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))


class Neterr:
    def __init__(self, inputdim, outputdim,inputarr, rng, hidden_unit_lim):
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
    def feedforwardcm(self):
        ConnMatrix = {}
        WeightMatrix = {}
        NatureCtrDict = {}
        NatureCtrDict['I'] = 0
        NatureCtrDict['H1'] = 0
        NatureCtrDict['H2'] = 0
        NatureCtrDict['O'] = 0
        
        dictionary = {}
        dictionary['I'] = {}
        dictionary['H1'] = {}
        dictionary['H2'] = {}
        dictionary['O'] = {}
        age = -1
        chromo = Chromosome(age)

        for i in chromo.node_arr:
            dictionary[i.nature][i.node_num] = NatureCtrDict[i.nature]
            NatureCtrDict[i.nature] += 1
        
        ConnMatrix['IO'] = np.zeros((inputdim, outputdim))
        ConnMatrix['IH1'] = np.zeros((inputdim, H1))
        ConnMatrix['IH2'] = np.zeros((inputdim, H2))
        ConnMatrix['H1H2'] = np.zeros((H1, H2))
        ConnMatrix['H1O'] = np.zeros((H1, outputdim))
        ConnMatrix['H2O'] = np.zeros((H2, outputdim))

        WeightMatrix['IO'] = np.zeros((inputdim, outputdim))
        WeightMatrix['IH1'] = np.zeros((inputdim, H1))
        WeightMatrix['IH2'] = np.zeros((inputdim, H2))
        WeightMatrix['H1H2'] = np.zeros((H1, H2))
        WeightMatrix['H1O'] = np.zeros((H1, outputdim))
        WeightMatrix['H2O'] = np.zeros((H2, outputdim))

        for con in chromo.conn_arr:
            ConnMatrix[con.source.nature + con.destination.nature][dictionary[con.source.nature][con.source.node_num]][dictionary[con.destination.nature][con.destination.node_num]] = 1
            WeightMatrix[con.source.nature + con.destination.nature][dictionary[con.source.nature][con.source.node_num]][dictionary[con.destination.nature][con.destination.node_num]] = con.weight 



        
    def feedforward(self):

        conn_list = priortize_connections(chromosome[conn_arr])
        storage = [0 for i in range(self.hidden_unit_lim + self.outputdim)]
        for i in range(self.inputarr.shape[0]):
            storage=[0]+list(self.inputarr[i]).append(storage) #here [0] is dummy storage as we use '1' indexing for node_num

        return np.array(lis)
        #pass

    def test(self, weight_arr):
        pass

    def modify_thru_backprop(self, popul, epochs=10, learning_rate=0.01, L1_reg=0.00001, L2_reg=0.0001):
        pass


def squa_test(x):
    return (x ** 2).sum(axis=1)


def main():
    # print("hi")
    import copy
    """trainarr = np.concatenate((np.arange(0,9).reshape(3,3),np.array([[1,0],[0,1],[1,0]])),axis=1)
    testarr = copy.deepcopy(trainarr)
    trainx=trainarr[:,:3]
    trainy=trainarr[:,3:]
    testx=testarr[:,:3]
    testy=testarr[:,3:]
    hid_nodes = 4
    indim = 3
    outdim = 2
    size = 5
    """
    hid_nodes = 4
    indim = 10
    outdim = 1
    size = 100
    resularr = np.zeros((size, outdim))
    for i in range(size):
        # resularr[i][np.random.randint(0,outdim)]=1
        if np.random.randint(0, 2) == 1:
            resularr[i][0] = 1
    # resularr
    trainarr = np.concatenate((np.arange(0, 1000).reshape(100, 10), resularr), axis=1)
    testarr = copy.deepcopy(trainarr)
    trainx = trainarr[:, :indim]
    trainy = trainarr[:, indim:]
    testx = testarr[:, :indim]
    testy = testarr[:, indim:]
    # arr_of_net = np.random.uniform(-1,1,(size,(indim+1)*hid_nodes+(hid_nodes+1)*outdim))
    hid_nodesarr = np.random.randint(1, hid_nodes + 1, size)
    lis = []
    for i in hid_nodesarr:
        lis.append(np.concatenate((np.array([i]), np.random.uniform(-1, 1, (indim + 1) * i + (i + 1) * outdim))))
    arr_of_net = np.array(lis)
    neter = Neterr(indim, outdim, arr_of_net, trainx, trainy, testx, testy)
    Backnet(3, neter)
    print(neter.trainx, neter.trainy)
    print(arr_of_net)
    print(neter.feedforward())
    for i in range(size):
        print(neter.test(arr_of_net[i]))


if __name__ == '__main__':
    main()
