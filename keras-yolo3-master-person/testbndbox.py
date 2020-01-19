import cv2
import numpy as np
#
# options = {"model":"cfg/yolo-1c.cfg","load":-1, "threshold": 0,"gpu":0.8}
# options = {"pbLoad":"built_graph\\yolov2-tiny.pb","metaLoad":"built_graph\\yolov2-tiny.meta", "threshold": 0,"gpu":0.8}
# tfnet = TFNet(options)

imgcv = cv2.imread("000000120021.jpg")

# result = tfnet.return_predict(imgcv)
#
# print(result)

def boxing(original_img):
    newImage = np.copy(original_img)


    top_x=int(276)
    top_y = int(0)
    btm_x = int(337)
    btm_y=int(118)


    newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (0, 0, 255), 2)

    return newImage

import matplotlib.pyplot as plt
cv2.imshow("f", boxing(imgcv))
cv2.waitKey(0)
# _, ax = plt.subplots(figsize=(20, 10))
# ax.imshow(boxing(imgcv, result))
# plt.show()