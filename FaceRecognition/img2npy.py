import os
from PIL import Image as img
import numpy as np
from matplotlib.pyplot import plot as plt
import cv2

imgData = None
Y = []
for roots,dirs,files in sorted(os.walk("../Face/data")):
    print(roots)
    for file in files:
        im = cv2.imread(roots + "/" + file,cv2.IMREAD_GRAYSCALE)
        print("Read file :",file)
        imresize = cv2.resize(im.copy(), dsize=(48, 48))
        imresize = imresize.reshape(1, imresize.shape[0], imresize.shape[1], 1)

        if imgData is None:
            imgData = imresize
        else:
            imgData = np.vstack((imgData,imresize))

        label = roots.split("\\")[1]
        print("Read label :",label)
        Y.append(label)

Y = np.array(Y)
Y = Y.reshape(Y.shape[0],1)

print(imgData.shape)
print(Y.shape)

np.save('imgData',imgData)
np.save("imgLabel",Y)





# a = np.array([1,2,3,4,5])
# b = np.array([5,3,2,5,1])
# c = np.vstack((a,b))
#
# print(c)


