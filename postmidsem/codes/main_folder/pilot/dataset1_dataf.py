#dataset1_dataf.py
'''
	this contains utilities function to extract and preprocess data from dataset1 ( with reference to all the datasets that is
	suggested in a mail from rkala dated Jan 2 2018  .) This dataset has two domains, namely, source and target and these two use the same objects and task
	except that target has fewer examples and is recorded in a different context. source_features.xlsx and target_features.xlsx are table with 200 X 640 and 50 X 640.
	there are total 10 objects recorded.
'''

import pandas as pd
import random
import numpy as np
rng = random
#to convert .xlsx to feature array for source
source_data = pd.read_excel("./dataset_1/source_features.xlsx", header=None)
source_feat_mat = source_data.as_matrix() 
assert( source_feat_mat.shape[0] == 200 )
#to read the labels for source
file_ob = open( "./dataset_1/source_output.csv", "rb")
lis = file_ob.readlines()
lis = lis[1:]
assert( len(lis) == 200)
source_label_lis = [item.rstrip().lower().decode("utf-8") for item in lis]
ann = len(lis)
file_ob.close()

#converting label in string to that to corresponding number
sorted_lis = list(set( source_label_lis ))
sorted_lis.sort()
dic = { sorted_lis[i]:i for i in range(len(sorted_lis)) }
print(dic)
source_label_lis_num = [ dic[item] for item in source_label_lis ]
source_label_lis_num_arr = np.array( source_label_lis_num )
dum_arr = source_label_lis_num_arr.reshape((source_label_lis_num_arr.shape[0], 1))
clumped_arr = np.concatenate( (source_feat_mat, dum_arr), axis = 1)
numlis = np.arange(clumped_arr.shape[0])
print(clumped_arr[:3])
rng.shuffle(numlis)
clumped_arr = clumped_arr[ numlis ]
clumped_source = clumped_arr[:]
 

#same like above for target
target_data = pd.read_excel("./dataset_1/target_features.xlsx", header = None)
target_feat_mat = target_data.as_matrix()
assert( target_feat_mat.shape[0] == 50 )
#to read the labels for source
file_ob = open( "./dataset_1/target_output.csv", "rb")
lis = file_ob.readlines()
lis = lis[1:]
assert( len(lis) == 50)
target_label_lis = [item.rstrip().lower().decode("utf-8") for item in lis]
bann = len(lis)
file_ob.close()

#converting label in string to that to corresponding number
sorted_lis = list(set( target_label_lis ))
sorted_lis.sort()
dic = { sorted_lis[i]:i for i in range(len(sorted_lis)) }
#print(dic)
target_label_lis_num = [ dic[item] for item in target_label_lis ]
target_label_lis_num_arr = np.array( target_label_lis_num )
dum_arr = target_label_lis_num_arr.reshape((target_label_lis_num_arr.shape[0], 1))
clumped_arr = np.concatenate( (target_feat_mat, dum_arr), axis = 1)
#print(dic)
numlis = np.arange(clumped_arr.shape[0])
rng.shuffle(numlis)
clumped_arr = clumped_arr[ numlis ]
#clumped_arr = clumped_arr[ numlis ]
clumped_target = clumped_arr[:]

#rng.shuffle(pimadata)

def give_source_data(): 
	'''
		returns two tuples each tuple has one array as feature set and other as column arrray of numerical labels
		this one is for source
	'''
	aann = (ann*3)//4
	#print(aann)
	rest_setx=clumped_source[:aann,:-1]#tuple of two shared variable of array

	rest_sety=clumped_source[:aann,-1:]
	test_setx=clumped_source[aann:,:-1]
	test_sety=clumped_source[aann:,-1:]
	assert( rest_setx.shape == (150,640))
	#print(rest_sety.shape)
	assert( rest_sety.shape == (150,1))
	assert( test_setx.shape == (50,640))
	assert( test_sety.shape == (50,1))
	return ((rest_setx,rest_sety),(test_setx,test_sety))


def give_target_data():  
	bbann = (bann*3)//4
	rest_setx=clumped_target[:bbann,:-1]
	rest_sety=clumped_target[:bbann,-1:]
	test_setx=clumped_target[bbann:,:-1]
	test_sety=clumped_target[bbann:,-1:]
	#print(test_setx.shape)
	#print(test_sety.shape)
	assert( rest_setx.shape == (37,640))
	assert( rest_sety.shape == (37,1))
	assert( test_setx.shape == (13,640))
	assert( test_sety.shape == (13,1))
	return ((rest_setx,rest_sety),(test_setx,test_sety))

def main():
	print( give_source_data()[0][0][:10])
	print( give_target_data()[1][0][:10])
if __name__=="__main__":
	main()
