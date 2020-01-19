import cv2
import numpy as np


imgcv = cv2.imread("1_0000003_0_0_0_6_aug_1.jpg")

def boxing(original_img):
    newImage = np.copy(original_img)


    top_x=int(714)
    top_y = int(209)
    btm_x = int(1010)
    btm_y=int(376)


    newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (0, 0, 255), 2)


    return newImage

cv2.imshow("f", boxing(imgcv))
cv2.waitKey(0)
