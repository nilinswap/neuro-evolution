import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
import pickle
import copy
import time
NGEN = 10000
pop_size = 100
cxpb = .8
m_fac = .2
m_prob = .2


np.random.seed(0)
rng = np.random
pstri = "./pickle_jar/"


fs = open(pstri + "tar_data_with_dic.pickle", "rb")
tup_t = pickle.load(fs)
fs.close()
target_arr, target_label,  target_dic = tup_t

dum_arr = target_label.reshape((target_label.shape[0], 1))
clumped_arr = np.concatenate((target_arr, dum_arr), axis=1)

# print(dic)
numlis = np.arange(clumped_arr.shape[0])
rng.shuffle(numlis)
clumped_arr = clumped_arr[numlis]
# clumped_arr = clumped_arr[ numlis ]
clumped_target = clumped_arr[:]
ann = int((3/4)*clumped_target.shape[0])
print(ann)
tup_t = (target_rest_arr, target_rest_label), (target_test_arr, target_rest_label) = (clumped_target[:ann, :-1], clumped_target[:ann, -1:]), (clumped_target[ann:, :-1], clumped_target[ann:, -1:])
#print(tup_t)
fs = open(pstri + "tar_tup.pickle", "wb")
pickle.dump(tup_t, fs)
fs.close()

target_dim = target_rest_arr.shape[1]


fs = open(pstri + "src_data_with_dic.pickle", "rb")
tup_s = pickle.load(fs)
fs.close()
source_arr, source_label, source_dic = tup_s


dum_arr = source_label.reshape((source_label.shape[0], 1))
clumped_arr = np.concatenate((source_arr, dum_arr), axis=1)
# print(dic)
numlis = np.arange(clumped_arr.shape[0])
rng.shuffle(numlis)
clumped_arr = clumped_arr[numlis]
# clumped_arr = clumped_arr[ numlis ]
clumped_source = clumped_arr[:]
#ann = (3//4)*clumped_source.shape[0]
ann  = clumped_source.shape[0]
tup_s = (source_rest_arr, source_rest_label), (source_test_arr, source_rest_label) = (clumped_source[:ann, :-1], clumped_source[:ann, -1:]), (clumped_source[ann:, :-1], clumped_source[ann:, -1:])
fs = open(pstri + "src_tup.pickle", "wb")
pickle.dump(tup_s, fs)
fs.close()
source_dim = source_rest_arr.shape[1]

def generate_pop(pop_size, source_dim, target_dim, rng = np.random):
	pop_lis = []
	for individual in range(pop_size):
		W = rng.random((source_dim, target_dim))
		pop_lis.append(W)
	return pop_lis

def dist(transformed_target, source_instance):
	return np.sqrt(np.sum((transformed_target - source_instance)**2))	

def closeness_cost(W):
	sumi = 0
	#print( source_dic)
	
	for class_num in target_dic:
		for target_instance in target_rest_arr[ target_dic[class_num][0]: target_dic[class_num][1]+1 ]:
			
			min_dist = np.inf
			target_instance = np.reshape(target_instance, (target_instance.shape[0], 1))
			transformed_target = np.dot(W, target_instance)
			for source_instance in source_rest_arr[source_dic[class_num][0]: source_dic[class_num][1]+1 ]:
				#print(transformed_target, source_instance)
				
				min_dist = min( min_dist, dist(transformed_target, source_instance))
			sumi += min_dist
	return sumi


def calc_fitness(population):
	cost_lis = []
	for indi in population:
		cost_lis.append(-closeness_cost(indi))
	return cost_lis

def myCrossover(arr1, arr2, cxpb, rng = np.random):

	for row in range(arr1.shape[0]):
		for col in range(arr1.shape[1]):
			if rng.random() < cxpb:
				alpha = rng.random()
				temp = copy.deepcopy(arr1[row][col])
				arr1[row][col] = alpha*arr1[row][col] + (1-alpha)*arr2[row][col]
				arr2[row][col] = alpha*arr2[row][col] + (1-alpha)*temp
	return arr1, arr2

def myMutate(arr, m_prob, m_fac, rng = np.random):
	arr =  arr + rng.random(arr.shape)*m_fac
	return arr

	for row in range(arr1.shape[0]):
		index = rng.randint(0, row)
		if rng.random() < m_prob: 
			arr[row][index] += rng.uniform(-1,1)*m_fac

def tournament_selection(population, fitness_arr, rng = np.random):
	a = rng.randint(0,len(population)-1)
	b = rng.randint(0,len(population)-1)
	parent1 = population[a]
	parent2 = population[b]

	if fitness_arr[a] < fitness_arr[b]:
		parentA = parent1
	else:
		parentA = parent2

	c = rng.randint(0,len(population)-1)
	d = rng.randint(0,len(population)-1)
	parent3 = population[c]
	parent4 = population[d]

	if fitness_arr[c] < fitness_arr[d]:
		parentB = parent3
	else:
		parentB = parent4

	return parentA, parentB

def main(pop_size):
	global source_dim, target_dim, m_fac, m_prob, cxpb
	population = generate_pop(pop_size, source_dim, target_dim)

	for i in range(NGEN):
		# print(population)
		print(i)
		fitness_arr = []   
		fitness_arr = calc_fitness(population)
		#print(fitness_arr)
		minn = np.inf
		if np.amin(fitness_arr) < minn:
			minn = np.amin(fitness_arr)
			ind_min = np.argmin(fitness_arr)

		print('minimum in this generation is '+ str(np.amin(fitness_arr)), "at", ind_min, "th index")
		
		mating_pool = population
		
		for j in range(int(pop_size/2)):
			parent1, parent2 = tournament_selection(population, fitness_arr)	
			child1, child2 = myCrossover(parent1, parent2, cxpb)
			child1 = myMutate(child1, m_fac, m_prob)
			child2 = myMutate(child2, m_fac, m_prob)
	ind_min = np.argmin(fitness_arr)
	print(population[ ind_min ])

	fs= open("dublue.pickle", "wb")
	pickle.dump(population[ind_min], fs)
	fs.close()

if __name__ == "__main__":
	main(pop_size)