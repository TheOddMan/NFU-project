import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

def mkdir(path):

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
         # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

if __name__ == "__main__":

    IMG_DIR = "D:\\XinYu\\NFU\\CaptureHandPackage\\HandData\\5"

    AUG_IMG_DIR = "AUG_IMG" # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)

    AUGLOOP = 5 # 每张影像增强的数量


    seq = iaa.Sequential([
        iaa.Add([-50,-20,20,50]),
        iaa.GaussianBlur([1.5,2,2.5]),
        iaa.MotionBlur([7,8,9,10]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        ])

    for root, sub_folders, files in os.walk(IMG_DIR):

        for name in files:

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                # 读取图片
                try:
                    img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.png'))
                except:
                    img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                img = np.array(img)

                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                Image.fromarray(image_aug).save(path)



