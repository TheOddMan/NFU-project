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

def recHand(imageHand,hand_rec_model):

    hand_gesture = ['0', '1', '2', '3', '4', '5']
    try:
        im_hand_gray = cv2.cvtColor(imageHand, cv2.COLOR_RGB2GRAY)
        im_hand_gray_resize = cv2.resize(im_hand_gray.copy(), dsize=(48, 48))
        im_hand_gray_resize = im_hand_gray_resize.reshape(1, im_hand_gray_resize.shape[0], im_hand_gray_resize.shape[1],
                                                          1)
        results = hand_rec_model.predict_classes(im_hand_gray_resize)
        print("Hand Result : ", hand_gesture[results[0]])

        return hand_gesture[results[0]]
    except:
        print("An exception occurred")


def detect_img(yolo,img,origin_person_img,hand_rec_model):

    img_or = img.copy()
    r_image,out_boxes = yolo.detect_image(img)

    r_image_array = np.array(r_image)
    out_boxes = np.round(out_boxes).astype("int")

    hand_List = []


    for box in out_boxes:
        h = box[2]-box[0]+50
        w = box[3]-box[1]
        y = box[0]-80
        x = box[1]
        imagePerson = np.array(img_or)
        hand_List.append("hand")
        imageHand = imagePerson[y:y + h, x:x + w] #PIL image
        handResult = recHand(imageHand,hand_rec_model)
        # displayImage(r_image_array,imageHand)

        cv2.rectangle(origin_person_img, (x, y), (x + w, y + h), (0, 255, 0, 2), 2)
        # key = cv2.waitKey(1) & 0xFF
        # cv2.imshow("Image", imagePerson)
        cv2.putText(origin_person_img, handResult, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)



    return hand_List

if __name__ == '__main__':
    from argparse import Namespace

    args = Namespace(model_path="model_data/ep081-loss10.058-val_loss12.472.h5", anchors_path='model_data/yolo_anchors.txt',
                     classes_path="model_data/hands_class.txt")

    detect_img(YOLO(**vars(args)),"a.png")

