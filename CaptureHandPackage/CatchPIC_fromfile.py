import cv2
import time
from imutils.video import VideoStream
import imutils
import numpy as np
from argparse import Namespace
from hand_detection.yolo import YOLO
from PIL import Image
from hand_detection.cropHand import detect_img




def load_handyolo(model_path,anchors_path='hand_detection/model_data/yolo_anchors.txt',classes_path="hand_detection/model_data/hands_class.txt"):
    args = Namespace(model_path=model_path,
                     anchors_path=anchors_path,
                     classes_path=classes_path)
    yoloForHand = YOLO(**vars(args))

    return yoloForHand




def CatchPICFromVideo(start_number=1,imagePrefix="train",imgFolder="HandData",fileName='',model="hand_detection/ep081-loss10.058-val_loss12.472.h5",confidenceV=0.5):
    print("[INFO] loading model...")
    hand_model = load_handyolo(model)

    print("[INFO] starting video stream...")
    vc = cv2.VideoCapture(fileName)
    time.sleep(2.0)
    num = start_number

    while (vc.isOpened()):
        ret, frameGBR = vc.read()
        frameGBR = imutils.resize(frameGBR, width=1400)
        (h, w) = frameGBR.shape[:2]
        frameRGB = frameGBR[:, :, ::-1]

        frameRGB_PIL = Image.fromarray(frameRGB, 'RGB')

        hands,_, outboxes = detect_img(hand_model, frameRGB_PIL)

        h,w,y,x = 0,0,0,0

        for box in outboxes:
            h = box[2] - box[0] + 50
            w = box[3] - box[1]
            y = box[0] - 50
            x = box[1]

            cv2.rectangle(frameGBR, (x, y), (x + w, y+h), (0, 0, 255), 2)

        cv2.imshow("Frame", frameGBR)

        imgname = imagePrefix + "_%08d.jpg" % (num)

        key = cv2.waitKey(1) & 0xFF

        crop_img = frameGBR[y:y + h, x:x + w]

        if key == 32:
            print("Capture image..", imgname)
            cv2.imwrite(imgFolder + "/" + imgname, crop_img)


            num += 1



        if key == ord('q'):
            print("quit..")
            break



CatchPICFromVideo(imagePrefix="5_",start_number=300,fileName='D:/XinYu/NFU/IMG_2210.mov')
