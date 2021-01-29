
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import Session
import tensorflow as tf
import math
import numpy as np

import pdb


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

class GenericNet:
    @staticmethod
    def build(width, height, depth, classes, 
              conv_depths=[8, 8],
              conv_pool=[True, True],
              conv_shape=(3,3),
              pool_shape=(2,2),
              pool_stride=(2,2),
              fc_depths=[4,4],
              batch_norm=False,
              model_type='sequential',
              add_DCC_layer=False,
              batch_size=None):
        if add_DCC_layer:
            model_type = 'functional'

        if K.image_data_format() == "channels_first":
            input_shape = [depth, height, width]
        else:
            input_shape = [height, width, depth]
        if batch_size is not None and model_type=='functional':
            input_shape = tuple([batch_size] + input_shape)

        if model_type == 'sequential':
            # initialize the model
            model = Sequential()
            # add convolutional layers
            first_depth = conv_depths[0]
            model.add(Conv2D(first_depth, conv_shape, 
                             padding="same",
                             activation="relu",
                             input_shape=input_shape))
            if conv_pool[0] is True:
                model.add(MaxPooling2D(pool_size=pool_shape, 
                                           strides=pool_stride))
            conv_depths = conv_depths[1:]
            conv_pool = conv_pool[1:]
            for (i, chann_depth) in enumerate(conv_depths):
                model.add(Conv2D(chann_depth, conv_shape, 
                                 padding="same",
                                 activation="relu"))
                if conv_pool[i] is True:
                    model.add(MaxPooling2D(pool_size=pool_shape, 
                                           strides=pool_stride))
            model.add(Flatten())

            # add fully connected layers
            for layer_depth in fc_depths:
                model.add(Dense(layer_depth, activation="relu"))

            # add classification layer      
            model.add(Dense(classes, activation="softmax"))

        else:
            if batch_size is not None:
                inp = Input(batch_shape=input_shape)
            else:
                inp = Input(shape=input_shape)
            x = inp
            for (i, chann_depth) in enumerate(conv_depths):
                x = Conv2D(chann_depth, conv_shape, 
                                 padding="same",
                                 activation="relu")(x)
                if conv_pool[i] is True:
                    x = MaxPooling2D(pool_size=pool_shape, 
                                           strides=pool_stride)(x)
            x = Flatten()(x)
            
            # add fully connected layers
            for (i,layer_depth) in enumerate(fc_depths):
                if i == len(fc_depths)-1 and add_DCC_layer:
                    x = Dense(layer_depth)(x)
                else:
                    x = Dense(layer_depth, activation="relu")(x)

            if add_DCC_layer:
                x = tf.expand_dims(x, 2)
                x = tf.matmul(x,x, transpose_b=True)
                ones = tf.ones(x.shape)
                mask = tf.linalg.band_part(ones, 0, -1)
                x = tf.boolean_mask(x, mask)
                # outdim is number of unique entries in correlation matrix,
                # which is just the number of unique entries in a symmetric matrix
                out_dim = int(nCr(fc_depths[-1],2) + fc_depths[-1])
                x = tf.reshape(x, [-1, out_dim])
                x = Flatten()(x)

            out = Dense(classes, activation="softmax")(x)
            model = Model(inp, out, name='genericnet')

        return model
