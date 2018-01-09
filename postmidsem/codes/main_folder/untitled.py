import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
import PIL

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


image = np.asarray(PIL.Image.open('Downloads/domain_adaptation_images/dslr/images/back_pack/frame_0001.jpg'))
image = to_gray(image)
print(image.shape)
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(250, 250),block_norm = 'L1-sqrt',
					cells_per_block=(1, 1), visualise=True)
print(fd.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
