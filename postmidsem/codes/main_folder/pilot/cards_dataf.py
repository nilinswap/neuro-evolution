import random
import numpy as np


# import pylab as pl
def standardize_dataset(traindata, means, stdevs):
	for row in traindata:
		for i in range(len(row)):

			row[i] = (row[i] - means[i])
			if stdevs[i]:
				row[i] /= stdevs[i]


rng = random



def get_dimension():
	in_dem = 8
	out_dem = 1
	return (in_dem, out_dem)


def myrange(start, end, step):
	i = start
	while i + step < end:
		i += step
		yield i


def load_data():
	def preprocessdata(dataset = "cards.data"):
		path = "./"
		import os.path
		filename = path + dataset
		if os.path.isfile(filename):
			#print("yes")
			fileo = open(filename, "r+")
			stlis = fileo.readlines()
			stlis = [i.rstrip().split(',') for i in stlis]

			mislis = []
			indlis = [i for i in range(len(stlis)) if '?' in stlis[i]]
			tempstlis = stlis[:]
			for i in indlis:
				mislis.append(tempstlis[i])
				stlis.remove(tempstlis[i])
			del (tempstlis)

			featarr = np.array(stlis)

			# first feat starts
			featarr[np.where(featarr[:, 0] == 'a'), 0] = 0
			featarr[np.where(featarr[:, 0] == 'b'), 0] = 1
			# ends

			# second starts
			# ends

			# third starts
			# ends

			# fourth starts
			featarr[np.where(featarr[:, 3] == 'u'), 3] = 0
			featarr[np.where(featarr[:, 3] == 'y'), 3] = 1
			featarr[np.where(featarr[:, 3] == 'l'), 3] = 2
			featarr[np.where(featarr[:, 3] == 't'), 3] = 3
			# fourth ends

			# fifth starts
			featarr[np.where(featarr[:, 4] == 'g'), 4] = 0
			featarr[np.where(featarr[:, 4] == 'p'), 4] = 1
			featarr[np.where(featarr[:, 4] == 'gg'), 4] = 2
			# fifth ends

			# sixth starts
			featvarlis = ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff']
			ite = range(len(featvarlis))
			for i in ite:
				featarr[np.where(featarr[:, 5] == featvarlis[i]), 5] = i
			# sixth ends

			# seventh starts
			featvarlis = ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o']
			ite = range(len(featvarlis))
			for i in ite:
				featarr[np.where(featarr[:, 6] == featvarlis[i]), 6] = i
			# that too ends

			# eighth starts
			# eighth ends

			# nineth starts
			featarr[np.where(featarr[:, 8] == 'f'), 8] = 0
			featarr[np.where(featarr[:, 8] == 't'), 8] = 1
			# well this ends too

			# tenth starts
			featarr[np.where(featarr[:, 9] == 'f'), 9] = 0
			featarr[np.where(featarr[:, 9] == 't'), 9] = 1
			# why not end this as well?

			# eleventh starts
			# and ends at the same place

			# twelweth starts
			featarr[np.where(featarr[:, 11] == 'f'), 11] = 0
			featarr[np.where(featarr[:, 11] == 't'), 11] = 1
			# everything ends...except true love

			# thirteenth starts
			featarr[np.where(featarr[:, 12] == 'g'), 12] = 0
			featarr[np.where(featarr[:, 12] == 'p'), 12] = 1
			featarr[np.where(featarr[:, 12] == 's'), 12] = 2
			# what's the matter with you?

			# You know what? I won't even mention forteenth and fifteenth

			featarr[np.where(featarr[:, 15] == '-'), 15] = 0
			featarr[np.where(featarr[:, 15] == '+'), 15] = 1

			# normalizing
			# featarr[:,:15]=(featarr[:,:15]-np.mean(featarr[:,:15],axis=0))/(np.max(featarr[:,:15],axis=0)-np.min(featarr[:,:15],axis=0))

			featarr = featarr.astype(float)
			traindata = featarr
			means = traindata.mean(axis=0)

			stdevs = np.std(traindata, axis=0)

			# standardize dataset
			def standardize_dataset(traindata, means, stdevs):
				for row in traindata:
					for i in range(len(row)):

						row[i] = (row[i] - means[i])
						if stdevs[i]:
							row[i] /= stdevs[i]

			standardize_dataset(traindata[:, :15], means, stdevs)
			#print(featarr[:, 15])
			return traindata, mislis
		else:
			print("file could not be loaded")
	featarr, mislis = preprocessdata()  # here mislis is list type and featar is nd.array type
	numlis = np.arange(featarr.shape[0])
	rng.shuffle(numlis)
	#print(numlis)
	featarr = featarr[numlis]
	#np.random.shuffle(featarr)

	test_set = featarr[:133, :15], featarr[:133, 15]  # keeping test set aside
	rest_set = featarr[133:, :15], featarr[133:, 15]  # have to apply cross-validation on the rest
	return rest_set, test_set




# print(traindata)
def give_data():
	# 1. make iris.data in usable form
	# 2. make input set and output set out of it
	# 3. make setpool out of the dataset
	# 4. make pcn and train it
	# 5. test on validation and testing set

	rest_setx = pimadata[:538, :8]  # tuple of two shared variable of array
	rest_sety = pimadata[:538, 8:]
	test_setx = pimadata[538:, :8]
	test_sety = pimadata[538:, 8:]
	# print(pimadata.shape)
	# print(rest_setx.shape,test_setx.shape)
	return ((rest_setx, rest_sety), (test_setx, test_sety))


def main():
	print(load_data()[0])


if __name__ == "__main__":
	main()
