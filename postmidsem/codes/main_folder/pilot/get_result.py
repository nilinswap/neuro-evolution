import numpy as np
from scipy import stats
#single_log_without_bp.txt
#single_log_with_bp_without_clustring1_200.txt
#single_log_with_bp1_200.txt
#single_log_with_bp1_80.txt
#single_log_without_bp_80.txt
#single_log_with_bp_without_clustring1_80.txt
import random
def GetResult(st):

    name = "./log_folder/final_log_folder/" + st

    file_ob = open(name, "r+")

    stlis = file_ob.readlines()
    stlis = [ item.rstrip().split(' ') for item in stlis ]
    #print(len(stlis))
    stlis = random.sample( stlis, 15)
    result_lis = [[float(item[1]), float(item[2])] for item in stlis]
    result_arr = np.array(result_lis)

    print(result_arr)

    mean = np.mean(result_arr, axis = 0)

    stdevs = np.std(result_arr, axis=0)

    return mean, stdevs

def GiveTTestResult(st1, st2):
    name1 = "./log_folder/final_log_folder/" + st1
    name2 = "./log_folder/final_log_folder/" + st2

    file_ob = open(name1, "r+")
    stlis = file_ob.readlines()
    stlis = [item.rstrip().split(' ') for item in stlis]
    # print(len(stlis))
    stlis = random.sample(stlis, 15)
    result_lis = [[float(item[1]), float(item[2])] for item in stlis]
    result_arr1 = np.array(result_lis)

    file_ob = open(name2, "r+")
    stlis = file_ob.readlines()
    stlis = [item.rstrip().split(' ') for item in stlis]
    # print(len(stlis))
    stlis = random.sample(stlis, 15)
    result_lis = [[float(item[1]), float(item[2])] for item in stlis]
    result_arr2 = np.array(result_lis)


    arr1 = result_arr1[:, 1]
    arr2 = result_arr2[:, 1]
    #print(arr1, arr2)

    t_val, p_val = stats.ttest_rel(arr1, arr2)
    return t_val, p_val


print(GetResult("single_log_without_bp_80.txt"))
#print(GiveTTestResult("single_log_without_bp.txt", "single_log_with_bp_without_clustring1_80.txt"))

