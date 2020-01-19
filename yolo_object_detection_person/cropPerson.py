# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
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
from  face_detection.FaceRec_image import FaceRecognize
np.random.seed(42)


# 載入手偵測模型
def load_handyolo(model_path,anchors_path='hand_detection/model_data/yolo_anchors.txt',classes_path="hand_detection/model_data/hands_class.txt"):
    args = Namespace(model_path=model_path,
                     anchors_path=anchors_path,
                     classes_path=classes_path)
    yoloForHand = YOLO(**vars(args))

    return yoloForHand

#載入臉偵測模型
def load_faceyolo(yoloFaceProto,yoloFaceDetectModel):
    net = cv2.dnn.readNetFromCaffe(yoloFaceProto, yoloFaceDetectModel)
    return net

#載入人偵測模型
def load_personyolo(yoloPersonConfig,yoloPersonWeights):
    net = cv2.dnn.readNetFromDarknet(yoloPersonConfig, yoloPersonWeights)
    return net

#將原始圖片進行人物偵測，回傳座標等數值
def detectPerson(image,net):

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)

    net.setInput(blob)

    layerOutputs = net.forward(ln)

    return layerOutputs

#將輸出進行NMS處理，消除額外的框
def nmsPerson(layerOutputs,__confidence,__threshold):

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

    return idxs,boxes,confidences,classIDs

__image = "images/000000000113.jpg"

#hand model
hand_detect_model_path = "hand_detection/ep081-loss10.058-val_loss12.472.h5"
hand_model = load_handyolo(hand_detect_model_path)
#

#face_model
face_proto_path = "face_detection/deploy.prototxt.txt"
face_detect_model_path = "face_detection/face.caffemodel"
face_rec_model_path = load_model("face_detection/M_final.hdf5")
face_model = load_faceyolo(face_proto_path, face_detect_model_path)
#

#person model
person_namefile_path = "yolo-coco/coco_person.names"
person_weight_path = "yolo-coco/yolov3.weights"
person_cfg_path = "yolo-coco/yolov3.cfg"
__confidence = 0.5
__threshold = 0.3
LABELS = open(person_namefile_path).read().strip().split("\n")
#

##載入完整圖片

#此圖片會被畫上框框，不適合切割
image = cv2.imread(__image)
#複製一份完整圖片(用來切割)
image_or = image.copy()
#取得原始圖片長寬
(H, W) = image.shape[:2]

##

#載入人偵測模型
person_model = load_personyolo(person_cfg_path, person_weight_path)

#將原始圖片進行人物偵測，回傳座標等數值
layerOutputs = detectPerson(image, person_model)

#將輸出進行NMS處理，消除額外的框
idxs,boxes,confidences,classIDs = nmsPerson(layerOutputs,__confidence,__threshold)

#將完整圖片進行切除臉、手的部分
def cropImage(image,LABELS,idxs,boxes,confidences,classIDs):
    # 若有任何物件在此張圖片內
    if len(idxs) > 0:
        # 迭代所有物件
     for i in idxs.flatten():
         # 只挑出人物偵測
          if classIDs[i] != 0:
           continue

          # 人物框座標
          (x, y) = (boxes[i][0], boxes[i][1])
          # 人物框長寬
          (w, h) = (boxes[i][2], boxes[i][3])

          # 畫人物框
          # cv2.rectangle(image, (x, y), (x + w, y + h), [0,0,255], 2)

          ###                                                                   臉部偵測                                                                  ###
          # 切割人物的部分(色彩為BGR)，此圖會送入臉部偵測
          crop_img_array_to_faceDetect = image_or[y:y + h, x:x + w]
          # studentID = FaceRecognize(face_rec_model_path, face_model, crop_img_array_to_faceDetect)
          # print("學號 : ",studentID)
          ###                                                                   臉部偵測                                                                  ###

          ###                                                                   手部偵測                                                                  ###
          # 將切割人物的部分作色彩轉換(色彩為RGB)，此圖會送入手部偵測(因手部偵測需要RGB)
          crop_img_array_to_handDetect = crop_img_array_to_faceDetect[:, :, ::-1]
          # 將切割人物的部分(色彩為RGB)轉換成PIL圖片格式，此圖會送入手部偵測(因手部偵測需要PIL格式)
          crop_img_PIL_to_hand_Detect = Image.fromarray(crop_img_array_to_handDetect, 'RGB')

          hands = detect_img(hand_model, crop_img_PIL_to_hand_Detect)
          print("手 : ",hands)
         ###                                                                   手部偵測                                                                  ###



          # cv2.imshow("Image", crop_img)
          # cv2.waitKey(0)

          # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
          # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,255], 2)

cropImage(image,LABELS,idxs,boxes,confidences,classIDs)

# cv2.imshow("Image", image)
# cv2.waitKey(0)