import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
import PIL
import pickle

pstri = './'
fstri = '/home/placements2018/forgit/domain_adaptation_images/'
dir_lis = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle']


def files(path):
	for file in os.listdir(path):
		if os.path.isfile(os.path.join(path, file)):
			yield file


def do_avg(dep_ar):
	assert (dep_ar.shape[0] == 3)
	return (dep_ar[0] + dep_ar[1] + dep_ar[2]) // 2


def do_weighted_avg(dep_ar):
	assert (dep_ar.shape[0] == 3)
	G = dep_ar[0] * 0.299 + dep_ar[1] * 0.587 + dep_ar[2] * 0.114
	return G


def to_gray(image_ar):
	assert (len(image_ar.shape) == 3)
	grey_ar = np.zeros(image_ar.shape[:-1])
	for rownum in range(image_ar.shape[0]):
		for colnum in range(image_ar.shape[1]):
			grey_ar[rownum][colnum] = do_weighted_avg(image_ar[rownum][colnum])
	return grey_ar


def find_features_amazon(file_st):
	image = np.asarray(PIL.Image.open(file_st))
	image = to_gray(image)
	# print(image.shape)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(150, 150), block_norm='L1-sqrt',
						cells_per_block=(1, 1), visualise=True)
	return fd


def find_features_dslr(file_st):
	image = np.asarray(PIL.Image.open(file_st))
	image = to_gray(image)
	# print(image.shape)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(500, 500), block_norm='L1-sqrt',
						cells_per_block=(1, 1), visualise=True)
	return fd


def find_features_webcam(file_st):
	image = np.asarray(PIL.Image.open(file_st))
	image = to_gray(image)
	# print(image.shape)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(211, 211), block_norm='L1-sqrt',
						cells_per_block=(1, 1), visualise=True)
	return fd


def make_data_from_image_webcam(stri, dir_lis):
	lislis = []
	label_lis = []
	pickle_dic = {}
	sumi = -1
	for dirnum, dir_st in enumerate(dir_lis):

		new_dir_stri = stri + dir_st + '/'
		file_lis = list(files(new_dir_stri))
		pickle_dic[dirnum] = (sumi+1, sumi+len(file_lis))
		sumi += len(file_lis)
		lis = []
		print(file_lis)

		for file_st in file_lis:
			fd_ar = find_features_webcam(new_dir_stri + file_st)
			lis.append(list(fd_ar))
			label_lis.append(dirnum)
		lislis += lis
	oned_ar = np.array(label_lis, dtype='float64')
	twod_ar = np.array(lislis, dtype='float64')
	assert (twod_ar.shape[0] == oned_ar.shape[0])
	return twod_ar, oned_ar, pickle_dic


def make_data_from_image_amazon(stri, dir_lis):
	lislis = []
	label_lis = []
	pickle_dic = {}
	sumi = -1
	for dirnum, dir_st in enumerate(dir_lis):
		new_dir_stri = stri + dir_st + '/'
		file_lis = list(files(new_dir_stri))
		lis = []
		print(file_lis)
		pickle_dic[dirnum] = (sumi + 1, sumi + len(file_lis))
		sumi += len(file_lis)
		for file_st in file_lis:
			fd_ar = find_features_amazon(new_dir_stri + file_st)
			lis.append(list(fd_ar))
			label_lis.append(dirnum)
		lislis += lis
	oned_ar = np.array(label_lis, dtype='float64')
	twod_ar = np.array(lislis, dtype='float64')
	assert (twod_ar.shape[0] == oned_ar.shape[0])
	return twod_ar, oned_ar, pickle_dic


def make_data_from_image_dslr(stri, dir_lis):
	lislis = []
	label_lis = []
	pickle_dic = {}
	sumi = -1
	for dirnum, dir_st in enumerate(dir_lis):
		new_dir_stri = stri + dir_st + '/'
		file_lis = list(files(new_dir_stri))
		lis = []
		print(file_lis)
		pickle_dic[dirnum] = (sumi + 1, sumi + len(file_lis))
		sumi += len(file_lis)

		for file_st in file_lis:
			fd_ar = find_features_dslr(new_dir_stri + file_st)
			lis.append(list(fd_ar))
			label_lis.append(dirnum)
		lislis += lis
	oned_ar = np.array(label_lis, dtype='float64')
	twod_ar = np.array(lislis, dtype='float64')
	assert (twod_ar.shape[0] == oned_ar.shape[0])
	return twod_ar, oned_ar, pickle_dic


def make_source_data():
	global fstri, dir_lis
	stri = fstri + 'amazon/images/'

	tup = make_data_from_image_amazon(stri, dir_lis)
	fs = open(pstri + "pickle_jar/src_data_with_dic.pickle", "wb")
	pickle.dump(tup, fs)
	fs.close()


def make_target_data():
	global fstri, dir_lis
	stri = fstri + 'dslr/images/'

	tup = make_data_from_image_dslr(stri, dir_lis)
	fs = open(pstri + "pickle_jar/tar_data_with_dic.pickle", "wb")
	pickle.dump(tup, fs)
	fs.close()

def make_all_pickles():
	pass

if __name__ == '__main__':
	make_source_data()
	make_target_data()
