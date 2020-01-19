import os
from PIL import Image as img
import numpy as np
import cv2

#
imgData = None
Y = []
for roots,dirs,files in sorted(os.walk("D:\\XinYu\\NFU\\CaptureHandPackage\\HandData_aug")):
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

        label = roots.split("\\")[-1]
        print("Read label :",label)
        Y.append(label)

Y = np.array(Y)
Y = Y.reshape(Y.shape[0],1)

print(imgData.shape)
print(Y.shape)

np.save('imgData',imgData)
np.save("imgLabel",Y)




#
# imgData = np.load("imgData.npy")
# Y = np.load("imgLabel.npy")
# print(imgData.shape)