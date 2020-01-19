from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from hand_detection.cropHand import detect_img
from argparse import Namespace
from hand_detection.yolo import YOLO
from PIL import Image
import numpy as np



def load_handyolo(model_path,anchors_path='hand_detection/model_data/yolo_anchors.txt',classes_path="hand_detection/model_data/hands_class.txt"):
    args = Namespace(model_path=model_path,
                     anchors_path=anchors_path,
                     classes_path=classes_path)
    yoloForHand = YOLO(**vars(args))

    return yoloForHand

hand_detect_model_path = "hand_detection/ep081-loss10.058-val_loss12.472.h5"
hand_model = load_handyolo(hand_detect_model_path)


# hands = detect_img(hand_model, crop_img_PIL_to_hand_Detect)


model = load_model('BestM.hdf5')

img = cv2.imread('D:\\XinYu\\NFU\\Hand\\A_3.jpg',cv2.IMREAD_GRAYSCALE) #BGR



imresize = cv2.resize(img.copy(), dsize=(48, 48))
imresizedisplay = cv2.resize(img.copy(), dsize=(46, 46))
# crop_img_array_to_handDetect = imresize[:, :, ::-1]

# crop_img_PIL_to_hand_Detect = Image.fromarray(crop_img_array_to_handDetect, 'RGB')
# hands,handnp = detect_img(hand_model, crop_img_PIL_to_hand_Detect)
# print(handnp)
# cv2.imwrite('color_img.jpg', cv2.cvtColor(handnp, cv2.COLOR_RGB2BGR))
imresize = imresize.reshape(1, imresize.shape[0], imresize.shape[1], 1)
#
result = model.predict_classes(imresize)
#
print(result)

# cv2.imshow('fsdf',imresize)

winname = "Test"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 1000,300)  # Move it to (40,30)
cv2.imshow(winname, imresizedisplay)
cv2.waitKey()
cv2.destroyAllWindows()