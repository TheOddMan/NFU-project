from __future__ import division
import sys
import scipy.io as sio
import cv2
from PIL import Image, ImageDraw, ImageFont
import xml.etree.cElementTree as ET
import numpy as np
import os, glob
import datetime
from lxml import etree



debug = False
running_from_path = os.getcwd()


# Based on https://stackoverflow.com/a/20679579
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


# Based on https://stackoverflow.com/a/20679579
def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


# https://arcpy.wordpress.com/2012/04/20/146/
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def write_xml(folder,image,image_name, boxes, savedir):
    height, width, depth = image.shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = image_name+".jpg"
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    for box in boxes:

        a = box[0][0][0][0]
        b = box[0][0][0][1]
        c = box[0][0][0][2]
        d = box[0][0][0][3]

        aXY = (a[0][1], a[0][0])
        bXY = (b[0][1], b[0][0])
        cXY = (c[0][1], c[0][0])
        dXY = (d[0][1], d[0][0])


        maxX = max(aXY[0], bXY[0], cXY[0], dXY[0])
        minX = min(aXY[0], bXY[0], cXY[0], dXY[0])
        maxY = max(aXY[1], bXY[1], cXY[1], dXY[1])
        minY = min(aXY[1], bXY[1], cXY[1], dXY[1])



        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = "hand"
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(minX)
        ET.SubElement(bbox, 'ymin').text = str(minY)
        ET.SubElement(bbox, 'xmax').text = str(maxX)
        ET.SubElement(bbox, 'ymax').text = str(maxY)



    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir,image_name+".xml")
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

def writeAnnotationFiles(set_name, root_path, write_image_boxes=False, save_images_with_boxes=False, show_image=False):
    firstFile = True

    processMax = 100000
    processed = 0

    images_dir = "images/"
    annotation_dir = "annotations/"
    new_annotations = "new_annotationsXML/"
    new_images = "new_images/"

    if not os.path.exists(new_annotations):
        os.makedirs(new_annotations)

    if save_images_with_boxes and not os.path.exists(new_images):
        os.makedirs(new_images)

    # Change directory to annotations to retreive only file names
    os.chdir(annotation_dir)
    matlab_annotations = glob.glob("*.mat")

    os.chdir(running_from_path)


    for file in matlab_annotations:
        firstBox = True
        if (processed >= processMax):
            break

        filename = file.split(".")[0]

        if (debug):
            print(file)

        content = sio.loadmat(annotation_dir + file, matlab_compatible=False)

        boxes = content["boxes"]

        p = cv2.imread(images_dir+filename+".jpg")

        write_xml("D:\\XinYu\\darkflow\\training_dataset\\training_data\\images",
                  p, filename, boxes.T, "new_annotationsXML")


start_time = datetime.datetime.now()

if (len(sys.argv) > 1):
    # training_config = {
    #     "test": "hand_dataset/test_dataset/test_data",
    #     "train": "hand_dataset/training_dataset/training_data",
    #     "validation": "hand_dataset/validation_dataset/validation_data"
    # }

    training_config = {
        "train": "images"
    }

    for index in range(1, len(sys.argv)):
        if sys.argv[index] not in training_config:
            print("Not a valid config value, either use no arguments or on of the following: "),
            for key, _ in training_config.items():
                print(key),
        else:
            writeAnnotationFiles(sys.argv[index], training_config[sys.argv[index]])
else:
    print("else")
    # writeAnnotationFiles("test", "hand_dataset/test_dataset/test_data")
    writeAnnotationFiles("train", "images")
    # writeAnnotationFiles("validation", "hand_dataset/validation_dataset/validation_data")

end_time = datetime.datetime.now()
seconds_elapsed = (end_time - start_time).total_seconds()
print("It took {} to execute this".format(hms_string(seconds_elapsed)))