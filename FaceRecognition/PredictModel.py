from keras.applications import VGG16

from keras import models, Input
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.models import load_model,Sequential
from keras.optimizers import Adam,RMSprop,Adagrad
from keras.constraints import max_norm

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Flatten, Dense, Input
from xFunction.xPredict.xconfusion_matrix import plot_confusion_matrix


def predict_model(modelName,img):

    model = load_model(modelName)

    results = model.predict_classes(img)

    classes = ['10761116','10861121','40441125','40441141','40441144']
