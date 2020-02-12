import numpy as np
import argparse
import time
import cv2
import os
from PIL import Image
from argparse import Namespace
from hand_detection.cropHand import detect_img
from hand_detection.yolo import YOLO
from keras.models import load_model
from imutils.video import VideoStream
from face_detection.FaceRec_image import FaceRecognize
from requests import Session
from signalr import Connection
import json

np.random.seed(42)


# 載入手偵測模型
def load_handyolo(model_path, anchors_path='hand_detection/model_data/yolo_anchors.txt',
                  classes_path="hand_detection/model_data/hands_class.txt"):
    args = Namespace(model_path=model_path,
                     anchors_path=anchors_path,
                     classes_path=classes_path)
    yoloForHand = YOLO(**vars(args))

    return yoloForHand


# 載入臉偵測模型
def load_faceyolo(yoloFaceProto, yoloFaceDetectModel):
    net = cv2.dnn.readNetFromCaffe(yoloFaceProto, yoloFaceDetectModel)
    return net


# 載入人偵測模型
def load_personyolo(yoloPersonConfig, yoloPersonWeights):
    net = cv2.dnn.readNetFromDarknet(yoloPersonConfig, yoloPersonWeights)
    return net


# 將原始圖片進行人物偵測，回傳座標等數值
def detectPerson(image, net):
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    return layerOutputs


# 將輸出進行NMS處理，消除額外的框
def nmsPerson(layerOutputs, __confidence, __threshold, H, W):
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > __confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, __confidence,
                            __threshold)

    return idxs, boxes, confidences, classIDs


__image = "images/000000000113.jpg"


def cv2_window_setting(namedWindow, resizeWindow_h, resizeWindow_w, moveWindow_x, moveWindow_y, imshow):
    try:
        cv2.namedWindow(namedWindow, 0)
        cv2.resizeWindow(namedWindow, resizeWindow_h, resizeWindow_w)
        cv2.moveWindow(namedWindow, moveWindow_x, moveWindow_y)
        cv2.imshow(namedWindow, imshow)
    except:
        pass


# 將完整圖片進行切除臉、手的部分
def cropImage(image, idxs, boxes, classIDs, frame_count, signalrHub):
    image_or = image.copy()

    answersDic = {'answer1':0,'answer2':0,'answer3':0,'answer4':0}


    # 若有任何物件在此張圖片內
    if len(idxs) > 0:
        # 迭代所有物件
        for i in idxs.flatten():
            # 只挑出人物偵測
            if classIDs[i] != 0:  # 非人物物件就跳過
                continue

            # 人物框座標
            (x, y) = (boxes[i][0], boxes[i][1])
            # 人物框長寬
            (w, h) = (boxes[i][2], boxes[i][3])

            # 畫人物框
            if showImage:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255, 2.5), 2)
                cv2_window_setting("Frame : " + str(frame_count), 640, 480, 40, 30, image)
                cv2.waitKey(0)

            ###                                                                   臉部偵測                                                                  ###
            # 切割人物的部分(色彩為BGR)，此圖會送入臉部偵測
            crop_img_to_faceDetect = image_or[y:y + h, x:x + w]
            studentID = FaceRecognize(face_rec_model, face_det_model, crop_img_to_faceDetect, frame_count=frame_count,
                                      showImage=showImage)
            print("學號 : ", studentID)
            ###                                                                   臉部偵測                                                                  ###

            ###                                                                   手部偵測                                                                  ###
            # 將切割人物的部分作色彩轉換(色彩為RGB)，此圖會送入手部偵測(因手部偵測需要RGB)
            crop_img_array_to_handDetect = crop_img_to_faceDetect[:, :, ::-1]
            # 將切割人物的部分(色彩為RGB)轉換成PIL圖片格式，此圖會送入手部偵測(因手部偵測需要PIL格式)
            crop_img_PIL_to_hand_Detect = Image.fromarray(crop_img_array_to_handDetect, 'RGB')
            handResult = detect_img(hand_det_model, crop_img_PIL_to_hand_Detect, crop_img_to_faceDetect, hand_rec_model,
                                    frame_count=frame_count,
                                    showImage=showImage)  # crop_img_to_faceDetect為了使手部框與臉部框畫在同一張影像
            if handResult == "": handResult = "nohand"
            print("手勢 : ", handResult)
            print("*" * 60)#
            print("*" * 60)
            ###                                                                   手部偵測                                                                  ###

            if handResult == '1':
                answersDic['answer1'] += 1
            elif handResult == '2':
                answersDic['answer2'] += 1
            elif handResult == '3':
                answersDic['answer3'] += 1
            elif handResult == '4':
                answersDic['answer4'] += 1





        if showImage:
            cv2.destroyAllWindows()


    answersJson = json.dumps(answersDic)
    if (frame_count % 5 == 0):
        signalrHub.server.invoke("send", answersJson)



# person model
person_namefile_path = "yolo-coco/coco_person.names"
person_weight_path = "yolo-coco/yolov3.weights"
person_cfg_path = "yolo-coco/yolov3.cfg"
__confidence = 0.5
__threshold = 0.3
LABELS = open(person_namefile_path).read().strip().split("\n")
# 載入人偵測模型
person_model = load_personyolo(person_cfg_path, person_weight_path)
#

# hand model
hand_detect_model_path = "hand_detection/ep081-loss10.058-val_loss12.472.h5"
hand_det_model = load_handyolo(hand_detect_model_path)
hand_rec_model = load_model('hand_detection/BestM.hdf5')
#

# face_model
face_proto_path = "face_detection/deploy.prototxt.txt"
face_detect_model_path = "face_detection/face.caffemodel"
face_rec_model = load_model("face_detection/M_0989.hdf5")
face_det_model = load_faceyolo(face_proto_path, face_detect_model_path)
#


# Debug參數
showImage = False  # 顯示每一楨影像並暫停

print("[INFO] starting video stream...")
vc = cv2.VideoCapture("test_video.mp4")
frame_count = 1

with Session() as session:
    connection = Connection("http://140.130.36.46/signalr", session)
    signalrHub = connection.register_hub('signalrHub')
    connection.start()

    with connection:

        while (vc.isOpened()):

            ret, frame = vc.read()

            if frame is None:
                break

            (H, W) = frame.shape[:2]

            layerOutputs = detectPerson(frame, person_model)

            idxs, boxes, confidences, classIDs = nmsPerson(layerOutputs, __confidence, __threshold, H, W)

            cropImage(frame, idxs, boxes, classIDs, frame_count, signalrHub)

            frame_count += 1
