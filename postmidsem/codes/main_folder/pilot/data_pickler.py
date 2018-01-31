import random
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
import PIL
import pickle
pstri = './'
fstri = '/home/placements2018/forgit/Dataset3/'
dir_lis_src = [ '062.eiffel-tower', '216.tennis-ball', '065.elk', '207.swan', '034.centipede']
dir_lis_tar = ['245.windmill', '017.bowling-ball', '105.horse', '060.duck', '190.snake']
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


def find_features(file_st):
	desired_size = 500
	#file_st = '/home/placements2018/forgit/Dataset3/060.duck/060_0032.jpg'
	image = PIL.Image.open(file_st)
	print(image.size)

	old_size = image.size  # old_size[0] is in (width, height) format
	ratio = float(desired_size) / max(old_size)
	new_size = tuple([int(x * ratio) for x in old_size])
	image = image.resize(new_size, PIL.Image.ANTIALIAS)
	#image = image.resize(new_size, PIL.Image.ANTIALIAS)
	# create a new image and paste the resized on it
	new_im = PIL.Image.new("RGB", (desired_size, desired_size))
	new_im.paste(image, ((desired_size - new_size[0]) // 2,
						 (desired_size - new_size[1]) // 2))
	new_im.save(file_st)
	#image = new_im
	#print(image.size)
	image = np.asarray(PIL.Image.open(file_st))
	print(image.shape)
	if len(image.shape) != 2:
		image = to_gray(image)
	# print(image.shape)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(image.shape[0]//2, image.shape[1]//2), block_norm='L1-sqrt',
						cells_per_block=(1, 1), visualise=True)
	print(file_st)
	print(fd.shape)
	assert( fd.shape[0] == 32)
	return fd



def make_data_from_image(stri, dir_lis, num = None):
	lislis = []
	label_lis = []
	for dirnum, dir_st in enumerate(dir_lis):
		new_dir_stri = stri + dir_st + '/'
		file_lis = list(files(new_dir_stri))
		lis = []
		print(file_lis)
		file_lis = random.sample(file_lis, len(file_lis))
		if num is not None:
			file_lis = file_lis[:num]
		for file_st in file_lis:
			fd_ar = find_features(new_dir_stri + file_st)
			lis.append(list(fd_ar))
			label_lis.append(dirnum)
		lislis += lis
	oned_ar = np.array(label_lis, dtype='float64')
	twod_ar = np.array(lislis, dtype='float64')
	assert (twod_ar.shape[0] == oned_ar.shape[0])
	return twod_ar, oned_ar
def make_source_data():
	global fstri, dir_lis_src
	stri = fstri
	
	tup = make_data_from_image( stri, dir_lis_src )
	fs = open( pstri+"pickle_jar/src_data.pickle", "wb")
	pickle.dump( tup , fs)
	fs.close()


def make_target_data():
	global fstri, dir_lis_tar
	stri = fstri
	
	tup = make_data_from_image( stri, dir_lis_tar, num = 20 )
	fs = open( pstri+"pickle_jar/tar_data.pickle", "wb")
	pickle.dump( tup , fs)
	fs.close()
	
if __name__ == '__main__':
	make_source_data()
	make_target_data()

