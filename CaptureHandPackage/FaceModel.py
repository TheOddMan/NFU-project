from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace


vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

print(vgg_features.summary())