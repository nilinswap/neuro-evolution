# gene.py
dict_of_sm_so_far = { }# error! error! error!
curr_gen_no = 1

class Node:
    def __init__(self, node_num, nature):
        self.node_num = node_num
        self.nature = nature  # 'I','H1','H2','O'

    def pp(self):
        print("\n\tNode")
        print("node_num: %d, nature: %r " % (self.node_num, self.nature))
        print("--------")

class Conn:
    def __init__(self, innov_num, couple, weight, status=True):
        self.status = status  # determines enabled or disabled
        self.source = couple[0]  # a node object
        self.destination = couple[1]  # same as above
        self.weight = weight
        self.innov_num = innov_num

    def get_couple(self):
        return (self.source, self.destination)

    def set_weight(self, weight):
        self.weight = weight
    def pp(self):
        print("\n\nConn")
        print("innov_num:", self.innov_num)
        print("couples are")
        self.get_couple()[0].pp()
        self.get_couple()[1].pp()
        print("weight",self.weight,"status",self.status)
        print("-----------------")

class BiasConn(Conn):
    def __init__(self, out_node, weight):
        numb = -1
        nature = -1
        status = True
        Conn.__init__(self, -1, (Node(numb, nature), out_node), weight, status=status)


