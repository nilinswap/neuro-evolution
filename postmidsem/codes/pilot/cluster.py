import numpy as np

import random
def distance ( chromoA, chromoB):

    same_num = 0
    for con1, con2 in zip(chromoA.conn_arr, chromoB.conn_arr):
        if con1.innov_num == con2.innov_num:
            same_num += 1
        else:
            break
    total_len = len(chromoA.conn_arr) + len(chromoB.conn_arr)

    diff_sum = total_len - 2 * same_num
    return (-same_num + diff_sum)




def give_new_head(dic):

    new_cluster_head = []
    for key in dic.keys():
        st = dic[key]
        new_head = None
        minfreq = -10000000
        for element in st:
            map = {}
            for element2 in st:

                dis = distance(element, element2)
                if dis not in map.keys():
                    map[dis] = 1
                else:
                    map[dis] += 1
            maxx = max(map.values())
            if maxx > minfreq:
                minfreq = maxx
                new_head = element
        if minfreq == 1:
            new_head = random.choice(list(st))
        new_cluster_head.append(new_head)

    return new_cluster_head





def give_cluster_head(chromo_list, k):
    print("hi")
    predefined_iter = 5
    print(chromo_list)
    current_cluster_head_list = random.sample(chromo_list, k)
    #print(current_cluster_head_list)
    dic = { key : set([]) for key in current_cluster_head_list}

    min_head = None

    for iter in range(predefined_iter):
        dic = {key: set([]) for key in current_cluster_head_list}
        for chromo in chromo_list:

            min = 10000000000000

            for cluster_head in current_cluster_head_list:

                dist = distance(chromo, cluster_head)

                if min > dist:
                    min = dist
                    min_head = cluster_head
            dic[min_head].add(chromo)
        if iter == 0 or iter == 4:
            print(dic)
            print()
        current_cluster_head_list = give_new_head(dic)


    return current_cluster_head_list






if __name__ == 'main':
    test()


