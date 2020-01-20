import cv2
import time
from imutils.video import VideoStream
import imutils
import numpy as np
from keras.models import load_model
import keras.backend as back
from PIL import Image as im

back.set_learning_phase(0)


def cv2_window_setting(namedWindow,resizeWindow_h,resizeWindow_w,moveWindow_x,moveWindow_y,imshow):
    try:
        cv2.namedWindow(namedWindow, 0)
        cv2.resizeWindow(namedWindow, resizeWindow_h, resizeWindow_w)
        cv2.moveWindow(namedWindow, moveWindow_x, moveWindow_y)
        cv2.imshow(namedWindow, imshow)
    except:
        pass


def FaceRecognize(FaceModel,net,image,frame_count,confidenceV=0.5):

    image_for_drawing = image.copy()    #image為被切割出來的學生區塊，image_for_drawing為畫上方框的影像(此影像不會拿去做辨識，因為有被畫方框，可能會影響辨識)

    face = ['10761116','10861121','40441125','40441141','40441144']

    (h, w) = image.shape[:2]
    print(h)
    print(w)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    studentID_List = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < confidenceV:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")


        crop_img = image[startY:endY, startX:endX]  #從學生區塊裁切臉部區塊

        crop_img_for_drawing = crop_img.copy()
        
        cv2_window_setting("Face_Crop : " + str(frame_count), 440, 380, 1200, 30, crop_img_for_drawing)

        crop_img = cv2.resize(crop_img, dsize=(48, 48))



        crop_img = crop_img.reshape(1, crop_img.shape[0], crop_img.shape[1], crop_img.shape[2])



        r = FaceModel.predict_classes(crop_img)

        studentID = face[r[0]]
        studentID_List.append(studentID)
        text = "{}".format(studentID)


        y = startY - 10 if startY - 10 > 20 else startY + 20

        cv2.rectangle(image_for_drawing, (startX, startY), (endX, endY), (0, 0, 255, 2), 2)
        cv2.putText(image_for_drawing, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)


        cv2_window_setting("Person_Face_Frame : " + str(frame_count),440,380,700,30,image_for_drawing)
        cv2.waitKey(0)



    return studentID_List


if __name__ == '__main__':
    FaceModel = load_model("M_final.hdf5")


    FaceRecognize(FaceModel=FaceModel,fileName="000000000113.jpg")
