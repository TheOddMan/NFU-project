from keras.preprocessing import image
# import cv2
# import pandas as pd
import numpy as np
# from tqdm import tqdm_notebook




def resize_to_square(im,img_size):
    old_size = im.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = cv2.resize(im,(new_size[1],new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size-new_size[0]
    top,bottom = delta_h//2,delta_h-(delta_h//2)
    left,right = delta_w//2,delta_w-(delta_w//2)
    color = [0,0,0]
    new_im = cv2.copyMakeBorder(im,top,bottom,left,right,
                               cv2.BORDER_CONSTANT,value=color)
    return new_im

# def load_image(path,id,img_size,extension="None"):

    # id = id.replace('.png','')
    id = id.split(".")[0]

    image = cv2.imread(path+"/"+id+extension)

    new_image = resize_to_square(image,img_size)

    return new_image




def generatorPrepare(test_data_dir,img_size,batch_size):

    # datagen = image.ImageDataGenerator(1./255)
    datagen = image.ImageDataGenerator(1. / 255)

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,shuffle=False,
        class_mode="categorical")

    return generator

def csv2imagePrepare(csvName,imagePath,img_size,csvIDcolumn='ID',csvLabelcolumn='Label',extension='None'):
    train_df = pd.read_csv(csvName)
    ids = train_df[csvIDcolumn].values
    X = []
    Y = []

    if extension == "None":
        extensionN = input("請輸入你的圖片副檔名(Please enter your extension of the images) : ")
        if not extensionN.startswith("."):
            extensionN = "."+extensionN
    else:
        if not extension.startswith("."):
            extensionN = "."+extension
        else:
            extensionN = extension




    print("Load Image...")
    for id in tqdm_notebook(ids):
        try:
            im = load_image(imagePath + "/", id, img_size,extension=extensionN)
            X.append(im)
            ads = train_df[train_df[csvIDcolumn] == id][csvLabelcolumn].values[0]
            Y.append(ads)
        except Exception as e:
            print("圖片讀取有錯誤 : ", e)
            print("Error when loading images : ", e)
            pass
    print("Finish Loading Image...")


    X = np.array(X)
    X = X.astype("float32")
    X /= .255
    Y = np.array(Y)

    return X, Y