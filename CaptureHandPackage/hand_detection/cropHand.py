import sys
import argparse
from hand_detection.yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import cv2
from hand_detection.yolo3.utils import letterbox_image
from keras import backend as K
import os

def displayImage(person_array,hand_array):
    im_person_gbr = cv2.cvtColor(person_array, cv2.COLOR_RGB2BGR)
    im_hand_gbr = cv2.cvtColor(hand_array, cv2.COLOR_RGB2BGR)  # PIL image to cv2 image
    im_hand_gbr = cv2.resize(im_hand_gbr, (400, 400))
    cv2.imshow("Person", im_person_gbr)
    cv2.imshow("Hand", im_hand_gbr)
    cv2.waitKey(0)


def detect_img(yolo,img):

    img_or = img.copy()
    r_image,out_boxes = yolo.detect_image(img)

    r_image_array = np.array(r_image)
    out_boxes = np.round(out_boxes).astype("int")

    hand_List = []
    imageHand = None
    for box in out_boxes:
        h = box[2]-box[0]
        w = box[3]-box[1]
        y = box[0]
        x = box[1]
        imagePerson = np.array(img_or)
        hand_List.append("hand")
        imageHand = imagePerson[y:y + h, x:x + w] #PIL image

        # displayImage(r_image_array,imageHand)
    # return hand_List,imageHand

    return hand_List,imageHand,out_boxes

if __name__ == '__main__':
    from argparse import Namespace

    args = Namespace(model_path="model_data/ep081-loss10.058-val_loss12.472.h5", anchors_path='model_data/yolo_anchors.txt',
                     classes_path="model_data/hands_class.txt")

    detect_img(YOLO(**vars(args)),"a.png")

