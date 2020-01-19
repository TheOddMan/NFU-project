import cv2
import time
from imutils.video import VideoStream
import imutils
import numpy as np
from keras.models import load_model
import keras.backend as back
from PIL import Image as im



def load_handyolo(model_path,anchors_path='hand_detection/model_data/yolo_anchors.txt',classes_path="hand_detection/model_data/hands_class.txt"):
    args = Namespace(model_path=model_path,
                     anchors_path=anchors_path,
                     classes_path=classes_path)
    yoloForHand = YOLO(**vars(args))

    return yoloForHand

hand_detect_model_path = "hand_detection/ep081-loss10.058-val_loss12.472.h5"
hand_model = load_handyolo(hand_detect_model_path)

back.set_learning_phase(0)

def FaceRecognize(FaceModel,start_number=1,imagePrefix="train",imgFolder="FaceData",fileName="",
                      proto="deploy.prototxt.txt",model='face.caffemodel',confidenceV=0.5):
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto, model)
    print("[INFO] starting video stream...")
    vc = cv2.VideoCapture(fileName)
    time.sleep(2.0)
    num = start_number

    face = ['40441125','40441141','40441144']

    while (vc.isOpened()):
        ret, frame = vc.read()
        frame = imutils.resize(frame, width=1400)

        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        (startX, startY, endX, endY) = (0,0,0,0)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidenceV:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            crop_w = endX - startX
            crop_h = endY - startY

            crop_img = frame[startY:endY, startX:endX]

            crop_img = cv2.resize(crop_img,dsize=(48,48))
            crop_img = crop_img.reshape(1,crop_img.shape[0],crop_img.shape[1],crop_img.shape[2])
            crop_img = crop_img.astype("float") / 255.0
            r = FaceModel.predict_classes(crop_img)

            sid = face[r[0]]

            text = "{:.2f},{}".format(confidence * 100,sid)
            y = startY - 10 if startY - 10 > 20 else startY + 20
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 0, 2), 2)

            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # imgname = imagePrefix + "_%08d.jpg" % (num)

        key = cv2.waitKey(1) & 0xFF

        cv2.imshow("Frame", frame)

        # crop_w = endX - startX
        # crop_h = endY - startY
        #
        # crop_img = frame[startY:endY, startX:endX]


        # if key == 32:
        #     print("Capture image..", imgname)
        #     cv2.imwrite(imgFolder + "/" + imgname, crop_img)
        #     print("crop_w : ", crop_w)
        #     print("crop_h : ", crop_h)
        #
        #     num += 1



        if key == ord('q'):
            print("quit..")
            break


FaceModel = load_model("M_final.hdf5")


FaceRecognize(imagePrefix="40441125",start_number=420,FaceModel=FaceModel,fileName="me.mp4")
