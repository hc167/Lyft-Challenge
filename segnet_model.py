from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Dropout, Layer, Activation, Reshape, Permute
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization

from helper import *

def get_encoding_layers(kernel = (3, 3), pool_size = 2):
    return [
        Conv2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def get_decoding_layers(kernel = (3, 3), pool_size = 2):
    return[
        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(size=(pool_size,pool_size)),
    ]

def get_segnet_basic():
    segnet_basic = Sequential()
    segnet_basic.add(Layer(input_shape=(height//2, width//2, 3)))

    segnet_basic.encoding_layers = get_encoding_layers()
    for l in segnet_basic.encoding_layers:
        segnet_basic.add(l)

    segnet_basic.decoding_layers = get_decoding_layers()
    for l in segnet_basic.decoding_layers:
        segnet_basic.add(l)

    segnet_basic.add(Conv2D(classes, (1, 1)))
    Activation('relu'),

    segnet_basic.add(Reshape((height*width, classes), input_shape=(height, width, classes)))
    segnet_basic.add(Activation('softmax'))

    return segnet_basic

def get_full_encoding_layers(kernel = (3, 3), pool_size = 2):
    return [
        Conv2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def get_full_decoding_layers(kernel = (3, 3), pool_size = 2):
    return[
        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        Conv2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(size=(pool_size,pool_size)),
    ]

def get_segnet_full():
    segnet_basic = Sequential()
    segnet_basic.add(Layer(input_shape=(height//2, width//2, 3)))

    segnet_basic.encoding_layers = get_full_encoding_layers()
    for l in segnet_basic.encoding_layers:
        segnet_basic.add(l)

    segnet_basic.decoding_layers = get_full_decoding_layers()
    for l in segnet_basic.decoding_layers:
        segnet_basic.add(l)

    segnet_basic.add(Conv2D(classes, (1, 1)))

    segnet_basic.add(Reshape((height*width, classes), input_shape=(height, width, classes)))
    segnet_basic.add(Activation('softmax'))

    return segnet_basic
