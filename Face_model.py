import numpy as np 
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

def Face_model(input_shape):

    x_input = Input(shape=(input_shape))

    x = ZeroPadding2D((3,3))(x_input)

    #layer 1

    x = Conv2D(64, (3,3), strides=(2,2))(x)
    x=BatchNormalization(axis=1, momentum=0.99, epsilon=0.0001)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=3, strides=2)(x)

    #layer 2

    x = Conv2D(128, (3,3), strides=(2,2))(x)
    x = BatchNormalization(axis = 1, epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=3, strides=2)(x)

    x = ZeroPadding2D((1,1))(x)

    #layer 3

    x = Conv2D(192, (5,5), strides=(2,2))(x)
    x = BatchNormalization(axis = 1, epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size=3, strides=2)(x)

    #layer 4 (1x1 conv)

    x = Conv2D(128,(1,1),strides=1)(x)
    x = BatchNormalization(axis = 1, epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = Activation('relu')(x)
    
    x = Lambda(lambda  a: K.l2_normalize(a,axis=1))(x)

    F_model = Model(inputs = x_input, outputs = x)

    F_model.summary()

    return F_model