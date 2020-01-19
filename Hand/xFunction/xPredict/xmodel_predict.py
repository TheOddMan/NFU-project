from .xconfusion_matrix import plot_confusion_matrix
from .xdata_prepare import generatorPrepare,csv2imagePrepare
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt




def generator_predict(model, test_data_dir, img_size,batch_size):

    test_generator = generatorPrepare(test_data_dir=test_data_dir,img_size=img_size,batch_size=batch_size)

    test_generator.reset()

    probabilities = model.predict_generator(generator=test_generator, steps=test_generator.samples/batch_size,verbose=1)

    predictions = np.rint(probabilities)
    predictions = np.argmax(predictions, axis=1)

    for i in predictions:
        print("results : "+str(i))

    y = test_generator.classes
    class_names = np.unique(y)
    class_names = class_names.astype('str')
    acc_score = accuracy_score(y, predictions, normalize=True)


    plot_confusion_matrix(y, predictions, class_names,acc_score)


    plt.show()

def csv2image_predict(model,csvName,imagePath,img_size,batch_size=1,extension="None"):

    X,Y = csv2imagePrepare(csvName=csvName,imagePath=imagePath,img_size=img_size,extension=extension)

    predictions = []

    for item in X:
        item = item.reshape(1, item.shape[0], item.shape[1], item.shape[2])
        prediction = np.argmax(model.predict(item,batch_size=batch_size))
        print("Model Predictions : ",prediction)
        predictions.append(prediction)

    predictions = np.array(predictions)

    class_names = np.unique(Y)

    class_names = class_names.astype('str')

    acc_score = accuracy_score(Y, predictions, normalize=True)

    plot_confusion_matrix(Y, predictions, class_names, acc_score)

    plt.show()






