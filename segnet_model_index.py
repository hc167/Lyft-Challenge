from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Dropout, Layer, Activation, Reshape, Permute, Input
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Concatenate
from keras import backend as K

from helper import *

def get_encoding_layers(kernel = (3, 3), pad = 1, pool_size = 2):
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(64, kernel),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        Activation('relu'),
    ]

def get_decoding_layers(kernel = (3, 3), pad = 1, pool_size = 2):
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(64, kernel),
        BatchNormalization(),
        Activation('relu'),
    ]

def get_segnet_basic():
    segnet_basic = Sequential()
    segnet_basic.add(Layer(input_shape=(height, width, 3)))

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

def get_full_encoding_layers(kernel = (3, 3), pad = 1, pool_size = 2):
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(64, kernel),
        BatchNormalization(),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(64, kernel),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel),
        BatchNormalization(),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        Activation('relu'),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        Activation('relu'),
    ]

def get_full_decoding_layers(kernel = (3, 3), pad = 1, pool_size = 2):
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(512, kernel),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(256, kernel),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel),
        BatchNormalization(),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(128, kernel),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(64, kernel),
        BatchNormalization(),
        ZeroPadding2D(padding=(pad,pad)),
        Conv2D(64, kernel),
        BatchNormalization(),
    ]

def get_segnet_full():
    segnet_basic = Sequential()
    segnet_basic.add(Layer(input_shape=(height, width, 3)))

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

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]

def get_segnet_basic_index(kernel = (3, 3), pad = 1, pool_size = 2):

    input = Input(shape=(height, width, 3))
    
    # encoder
    conv1 = ZeroPadding2D(padding=(pad,pad))(input)
    conv1 = Conv2D(64, kernel)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv1, mask1 = MaxPoolingWithArgmax2D(pool_size=(pool_size, pool_size))(conv1)

    conv2 = ZeroPadding2D(padding=(pad,pad))(conv1)
    conv2 = Conv2D(128, kernel)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv2, mask2 = MaxPoolingWithArgmax2D(pool_size=(pool_size, pool_size))(conv2)

    conv3 = ZeroPadding2D(padding=(pad,pad))(conv2)
    conv3 = Conv2D(256, kernel)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv3, mask3 = MaxPoolingWithArgmax2D(pool_size=(pool_size, pool_size))(conv3)

    conv4 = ZeroPadding2D(padding=(pad,pad))(conv3)
    conv4 = Conv2D(512, kernel)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # decoder
    conv5 = ZeroPadding2D(padding=(pad,pad))(conv4)
    conv5 = Conv2D(512, kernel)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = MaxUnpooling2D(size=(pool_size,pool_size))([conv5, mask3])

    conv6 = ZeroPadding2D(padding=(pad,pad))(conv6)
    conv6 = Conv2D(256, kernel)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = MaxUnpooling2D(size=(pool_size,pool_size))([conv6, mask2])

    conv7 = ZeroPadding2D(padding=(pad,pad))(conv7)
    conv7 = Conv2D(128, kernel)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv8 = MaxUnpooling2D(size=(pool_size,pool_size))([conv7, mask1])

    conv8 = ZeroPadding2D(padding=(pad,pad))(conv8)
    conv8 = Conv2D(64, kernel)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    # add softmax
    softmax = Conv2D(classes, (1, 1), padding="valid")(conv8)
    softmax = BatchNormalization()(softmax)
    softmax = Reshape((height*width, classes), input_shape=(height, width, classes))(softmax)
    output = Activation('softmax')(softmax)

    return  Model(inputs=input, outputs=output, name="SegNet")
