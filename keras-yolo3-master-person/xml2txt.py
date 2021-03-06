import xml.etree.ElementTree as ET
import os


def getLine(filename,txtfile):
    tree = ET.parse(filename)
    root = tree.getroot()
    imageName = root.find("filename").text
    imagePath = "trainingData/coco_cocolighting/"+imageName
    BoxFormat = ""
    for object in root.findall("object"):
        bndbox = object.find("bndbox")

        xmin = str(int(float(bndbox.find("xmin").text)))
        ymin = str(int(float(bndbox.find("ymin").text)))
        xmax = str(int(float(bndbox.find("xmax").text)))
        ymax = str(int(float(bndbox.find("ymax").text)))

        BoxFormat = BoxFormat + xmin + "," +ymin+","+xmax+","+ymax+",0 "


    Line = imagePath+" "+BoxFormat
    print(Line)
    txtfile.write(Line+"\n")
    print("====================")

with open("train_coco_cocolighting.txt", "a") as myfile:
    count  =1
    for root, subdirs, files in os.walk("trainingAnnotation/coco_cocolighting_Annotation"):
        for file in files:
            count = count+1
            getLine(root+"/"+file,myfile)

    print(count)

# getLine()