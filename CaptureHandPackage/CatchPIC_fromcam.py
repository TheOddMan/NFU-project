import cv2
import time
from imutils.video import VideoStream
import imutils
import numpy as np



def CatchPICFromVideo(start_number=1,imagePrefix="train",imgFolder="FaceData",
                      proto="deploy.prototxt.txt",model='face.caffemodel',confidenceV=0.5):
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(proto, model)
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    num = start_number

    while True:
        frame = vs.read()
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
            text = "{:.2f}".format(confidence * 100)
            y = startY - 10 if startY - 10 > 20 else startY + 20
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 0, 2), 2)

            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        imgname = imagePrefix + "_%08d.jpg" % (num)

        key = cv2.waitKey(1) & 0xFF

        cv2.imshow("Frame", frame)

        crop_w = endX - startX
        crop_h = endY - startY

        crop_img = frame[startY:endY, startX:endX]

        if key == 32:
            print("Capture image..", imgname)
            cv2.imwrite(imgFolder + "/" + imgname, crop_img)
            print("crop_w : ", crop_w)
            print("crop_h : ", crop_h)

            num += 1



        if key == ord('q'):
            print("quit..")
            break



CatchPICFromVideo(imagePrefix="40441125",start_number=1)
