import keras
from keras_applications import densenet
from keras.models import Model
from keras.layers import (Input,Dense,Dropout,Average,Flatten,
                          AveragePooling2D,BatchNormalization,
                          GlobalAveragePooling2D,Concatenate)
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import efficientnet.keras as efn
# import efficientnet.tfkeras as efn
import numpy as np
tf.disable_v2_behavior()



nb_classes = 4
batch_size = 8
epoches = 100

def build_densenet121(input_shape):
    first_inp = Input(shape=input_shape, name='input_1')
    base_model1 = densenet.DenseNet121(weights = 'imagenet',
                                     include_top=False,
                                     backend=keras.backend,
                                     layers=keras.layers,
                                     models=keras.models,
                                     utils=keras.utils)(first_inp)
    output1 = GlobalAveragePooling2D()(base_model1)            #1024
    # output1 = Dense(32,kernel_regularizer=keras.regularizers.l2(0.01))(output1)
    model = Model(inputs = first_inp, outputs = output1)
    return model


def build_efficientNetB0(input_shape):
    #第二条输入通道
    second_inp = Input(shape=input_shape, name='input_2')
    base_model2 = efn.EfficientNetB0(include_top=False,weights='imagenet',
                                       backend=keras.backend,
                                       layers=keras.layers,
                                       models=keras.models,
                                      utils=keras.utils)(second_inp)
    output2 = GlobalAveragePooling2D()(base_model2)                           #1280
    # output2 = Dense(32,kernel_regularizer=keras.regularizers.l2(0.01))(output2)
    model = Model(inputs = second_inp, outputs = output2)
    return model


def ensemble(input_shape1,input_shape2,nb_classes):

    model1 = build_densenet121(input_shape1)
    model2 = build_efficientNetB0(input_shape2)

    input1  = model1.input
    input2 = model2.input
    out1 = model1.output
    out2 = model2.output
    # 将两个网络的结果融合起来
    merger = Concatenate(axis=-1)([out1,out2])
    # merger = keras.layers.Dropout(0.5)(merger)
    op = Dense(nb_classes, activation='softmax',kernel_initializer="he_normal")(merger)          #
    model = Model(inputs=[input1,input2], outputs=op, name='ensemble')

    return  model
