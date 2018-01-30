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
desired_size = 250000



desired_size = 500
file_st= '/home/placements2018/forgit/Dataset3/060.duck/060_0032.jpg'
image = PIL.Image.open(file_st)
old_size = image.size  # old_size[0] is in (width, height) format
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
image = image.resize(new_size, PIL.Image.ANTIALIAS)
image = image.resize(new_size, PIL.Image.ANTIALIAS)
# create a new image and paste the resized on it
new_im = PIL.Image.new("RGB", (desired_size, desired_size))
new_im.paste(image, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
image = new_im
image = np.asarray(image)
image = to_gray(image)
print(image.shape)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(image.shape[0]//2, image.shape[1]//2),block_norm = 'L1-sqrt',
					cells_per_block=(2,2), visualise=True)
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
