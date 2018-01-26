import numpy as np
import chromosome
import random
import copy
indim = 32
outdim = 5
def distance ( chromoA, chromoB):

    same_num = 0
    assert (chromoA is not None)
    assert (chromoB is not None)
    for con1, con2 in zip(chromoA.conn_arr, chromoB.conn_arr):
        if con1.innov_num == con2.innov_num:
            same_num += 1
        else:
            break
    total_len = len(chromoA.conn_arr) + len(chromoB.conn_arr)

    diff_sum = total_len - 2 * same_num
    return (-same_num + diff_sum)




def give_new_head(dic):
    #new_head = chromosome.Chromosome(indim, outdim)
    new_head = None
    new_cluster_head = []
    for key in dic.keys():
        st = dic[key]
        assert ( len(st) != 0)
        minfreq = -10000000
        for element in st:
            mp = {}
            for element2 in st:

                dis = distance(element, element2)
                if dis not in mp.keys():
                    mp[dis] = 1
                else:
                    mp[dis] += 1
            maxx = max(mp.values())
            if maxx > minfreq:
                minfreq = maxx
                new_head = element

        if minfreq == 1:
            new_head = random.choice(list(st))

        new_cluster_head.append(new_head)

    return new_cluster_head





def give_cluster_head(chromo_list, k):
    #print("hi")
    predefined_iter = 20
    #print(chromo_list)
    current_cluster_head_list = random.sample(chromo_list, k)
    #current_cluster_head_list = chromo_list[:k]
    #print(current_cluster_head_list)

    dic = { key : set([]) for key in current_cluster_head_list}



    for iter in range(predefined_iter):
        dic = {key: set([key]) for key in current_cluster_head_list}
        for chromo in chromo_list:
            #min_head = None
            minn = 10000000000000
            assert (len(current_cluster_head_list) != 0)
            for cluster_head in current_cluster_head_list:
                assert (chromo is not None)
                if chromo == cluster_head and len(current_cluster_head_list) != 1:
                    continue
                dist = distance(chromo, cluster_head)

                if minn > dist:
                    minn = dist
                    min_head = cluster_head
            dic[min_head].add(chromo)

        current_cluster_head_list = give_new_head(dic)


    return current_cluster_head_list






if __name__ == 'main':
    test()


