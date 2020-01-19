import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import cv2
from yolo3.utils import letterbox_image
from keras import backend as K
import os



def detect_img(yolo,img):

    try:
        image = Image.open(img)
        image_or = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image,out_boxes = yolo.detect_image(image)
        out_boxes = np.round(out_boxes).astype("int")
        # r_image.show()
        for box in out_boxes:
            h = box[2]-box[0]
            w = box[3]-box[1]
            y = box[0]
            x = box[1]
            imagenp = np.array(image_or)
            crop_img = imagenp[y:y + h, x:x + w]
            # im_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # cv2.imshow("",im_rgb)
            # cv2.waitKey(0)
        # crop_img = Image.fromarray(crop_img, 'RGB')
        # crop_img.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    from argparse import Namespace

    args = Namespace(model_path="model_data/ep081-loss10.058-val_loss12.472.h5", anchors_path='model_data/yolo_anchors.txt',
                     classes_path="model_data/hands_class.txt")

    detect_img(YOLO(**vars(args)),"a.png")

