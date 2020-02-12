import cv2
import time
from imutils.video import VideoStream
import imutils
import numpy as np
from argparse import Namespace
from PIL import Image





def CatchPICFromVideo(start_number=1,imagePrefix="train",imgFolder="screenShot",fileName=''):


    print("[INFO] starting video stream...")
    vc = cv2.VideoCapture(fileName)
    time.sleep(2.0)
    num = start_number

    while (vc.isOpened()):
        ret, frameGBR = vc.read()
        frameGBR = imutils.resize(frameGBR, width=1400)
        (h, w) = frameGBR.shape[:2]


        cv2.imshow("Frame", frameGBR)

        imgname = imagePrefix + "_%08d.jpg" % (num)

        key = cv2.waitKey(1) & 0xFF



        if key == 32:
            print("Capture image..", imgname)
            cv2.imwrite(imgFolder + "/" + imgname, frameGBR)


            num += 1



        if key == ord('q'):
            print("quit..")
            break



CatchPICFromVideo(imagePrefix="screenShot_",start_number=1,fileName='D:/XinYu/NFU/IMG_2210.mov')
