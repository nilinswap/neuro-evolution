import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

base = '/home/dexter/Desktop/Domain_Adaptation/Square/'
files = [f for f in listdir(base) if isfile(join(base, f))]



if __name__ == "__main__":
    print(files)
    for i in range(len(files)):
        img = cv2.imread(base+files[i],0)
        print(type(img))
        equ = cv2.equalizeHist(img)
        #res = np.hstack((img,equ)
        cv2.imwrite('/home/dexter/Desktop/Domain_Adaptation/Square_results/'+files[i],equ)
#cv2.imshow(equ)
