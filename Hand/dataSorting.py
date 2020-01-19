import os
from shutil import  copyfile


fromDir = "data/test/"

for root, subdirs, files in os.walk(fromDir):

    if not root == fromDir:
        continue

    for file in files:

        fileClass = file.split("_")[1][0]

        print(file)
        print("File Class : ",fileClass)

        copyfile(fromDir+file,fromDir+fileClass+"/"+file)
