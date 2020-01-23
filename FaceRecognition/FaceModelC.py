from keras.applications import VGG16

from keras import models, Input
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Flatten, Dense
from keras.models import load_model,Sequential
from keras.optimizers import Adam,RMSprop,Adagrad
from keras.constraints import max_norm
from keras.models import  load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Flatten, Dense, Input
from xFunction.xPredict.xconfusion_matrix import plot_confusion_matrix


from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_npy(filename,labelname):

    imgData = np.load(filename+".npy")
    Y = np.load(labelname+".npy")

    return imgData,Y

def one_hot(Y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded,onehot_encoder

def build_model(lr,nb_classes,print_sum=True):
    model = models.Sequential()
    model.add(Conv2D(512, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes,activation='softmax'))


    optimizer = Adam(lr=lr)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    if print_sum:
        model.summary()

    return model

def splitdata(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42,stratify=Y)
    return X_train, X_test, y_train, y_test

def train_model(X,Y,model,batch_size,nb_epochs,model_name,load_model_name):


    c1 = ModelCheckpoint(filepath=model_name, save_best_only=True, monitor='val_acc', mode='max')

    callbackList = [c1]

    if len(load_model_name) != 0:
        print("Load pretrained model !!")
        model = load_model(load_model_name)
    else:
        print("New model !!")

    history = model.fit(X,Y,validation_split=0.3,batch_size=batch_size,callbacks=callbackList,shuffle=True,epochs=nb_epochs)

    model.save("M_final.hdf5")


    return history

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def predict_model(modelName,data,realValue,onehot_encoder):

    model = load_model(modelName)

    results = model.predict_classes(data)

    realValue = onehot_encoder.inverse_transform(realValue)
    realValue = realValue.ravel().astype("int")
    classes = ['10761116','10861121','40441125','40441141','40441144']
    plot_confusion_matrix(realValue,results,classes=classes,normalize=False)
    plt.show()

batch_size=128
nb_epochs=10
model_name="BestM.hdf5"
Predict_Model_Name = 'M_final.hdf5'
load_model_name = "M_0984.hdf5"
batch_size = 100
lr = 0.000001
nb_classes = 5
nb_epochs = 200
val_split=0.3
img_size=(48,48)
train_data_dir = 'test'
print_model_summary=True

imgData,Y = load_npy("imgData","imgLabel")

Y_one_hot,onehot_encoder = one_hot(Y)
X_train, X_test, y_train, y_test = splitdata(imgData,Y_one_hot)

model = build_model(lr,nb_classes,print_sum=True)
history = train_model(X_train,y_train,model,batch_size,nb_epochs,model_name,load_model_name)
plot_history(history)

predict_model(Predict_Model_Name,X_test,y_test,onehot_encoder)













