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

def cv2_window_setting(namedWindow,resizeWindow_h,resizeWindow_w,moveWindow_x,moveWindow_y,imshow):

    try:
        cv2.namedWindow(namedWindow, 0)
        cv2.resizeWindow(namedWindow, resizeWindow_h, resizeWindow_w)
        cv2.moveWindow(namedWindow, moveWindow_x, moveWindow_y)
        cv2.imshow(namedWindow, imshow)
    except:
        pass

def detect_img(yolo,img,origin_person_img,hand_rec_model):

    img_for_drawing = img.copy()

    img_for_drawing = np.array(img_for_drawing)

    r_image,out_boxes = yolo.detect_image(img)

    out_boxes = np.round(out_boxes).astype("int")

    hand_List = []


    for box in out_boxes:
        h = box[2]-box[0]+50
        w = box[3]-box[1]
        y = box[0]-80
        x = box[1]

        imagePerson = np.array(img) #imagePerson : RGB (PIL)

        hand_List.append("hand")

        imageHand = imagePerson[y:y + h, x:x + w] #PIL image

        imageHand_display = np.array(imageHand)

        cv2_window_setting("Hand_crop", 320, 240, 1200, 550, imageHand_display[:, :, ::-1])

        handResult = recHand(imageHand,hand_rec_model)

        cv2.rectangle(img_for_drawing, (x, y), (x + w, y + h), (0, 255, 0, 2), 2)

        cv2.putText(img_for_drawing, handResult, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2_window_setting("Person_Hand", 440, 380, 700, 450, img_for_drawing[:, :, ::-1])

    if len(hand_List) == 0:
        cv2.destroyWindow("Person_Hand")
        cv2.destroyWindow("Hand_crop")
        white_img_1 = np.full((320, 240), 255)
        white_img_2 = np.full((440, 380), 255)
        cv2_window_setting("Person_Hand", 440, 380, 700, 450, white_img_2)
        cv2_window_setting("Hand_crop", 320, 240, 1200, 550, white_img_1)

    return hand_List

if __name__ == '__main__':
    from argparse import Namespace

    args = Namespace(model_path="model_data/ep081-loss10.058-val_loss12.472.h5", anchors_path='model_data/yolo_anchors.txt',
                     classes_path="model_data/hands_class.txt")

    detect_img(YOLO(**vars(args)),"a.png")

