import json
from collections import defaultdict
import cv2
import xml.etree.ElementTree as ET
from lxml import etree
import os

name_box_id = defaultdict(list)
id_name = dict()
f = open(
    "mscoco2017/annotations/instances_train2017.json",
    encoding='utf-8')
data = json.load(f)

annotations = data['annotations']
count = 0

def write_xml(folder,width,height,depth,image_name, box_infos, savedir):


    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = image_name+".jpg"
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = "person"
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(x_min)
        ET.SubElement(bbox, 'ymin').text = str(y_min)
        ET.SubElement(bbox, 'xmax').text = str(x_max)
        ET.SubElement(bbox, 'ymax').text = str(y_max)



    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir,image_name+".xml")
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

for ant in annotations:
    id = ant['image_id']
    name = 'mscoco2017/train2017/%012d.jpg' % id
    cat = ant['category_id']

    if cat != 1:
        continue
    count = count+1

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])
print(len(name_box_id))
# f = open('train.txt', 'w')
# for key in name_box_id.keys():
#     f.write(key)
#     box_infos = name_box_id[key]
#     for info in box_infos:
#         x_min = int(info[0][0])
#         y_min = int(info[0][1])
#         x_max = x_min + int(info[0][2])
#         y_max = y_min + int(info[0][3])
#
#         box_info = " %d,%d,%d,%d,%d" % (
#             x_min, y_min, x_max, y_max, int(info[1]))
#         f.write(box_info)
#     f.write('\n')
# f.close()
folder = "D:/XinYu/NFU/keras-yolo3-master-person/mscoco2017/train2017"
savedir = "mscoco2017/annotationXML"
count = 0
for key in name_box_id.keys():
    box_infos = name_box_id[key]

    img = cv2.imread(key)
    height, width, channels = img.shape
    key = key.split("/")[-1].split(".")[0]
    print(key)
    write_xml(folder,width,height,channels,key,box_infos,savedir)




