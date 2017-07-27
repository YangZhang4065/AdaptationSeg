#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv2D,Input,Activation,MaxPooling2D,Dropout
from keras.layers import Cropping2D,Conv2DTranspose,Lambda
from keras.layers.merge import add

   
def create_vgg16_FCN(nb_rows, nb_cols,class_num):
    
    input_node = Input(shape=(3, nb_rows, nb_cols))
    # shape = (3, nb_rows, nb_cols)
    cnv1 = Conv2D(filters=64, kernel_size=(3, 3),  padding='same')(input_node)
    cnv1 = Activation('relu')(cnv1)
    
    cnv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(cnv1)
    cnv2 = Activation('relu')(cnv2)
    
    pool1 = MaxPooling2D((2, 2))(cnv2)
    # shape = (64, nb_rows/2, nb_cols/2)
    
    cnv3 = Conv2D(filters=128, kernel_size=(3, 3),padding='same')(pool1)

    cnv3 = Activation('relu')(cnv3)
    cnv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(cnv3)

    cnv4 = Activation('relu')(cnv4)
    pool2 = MaxPooling2D((2, 2))(cnv4)
    # shape = (128, nb_rows/4, nb_cols/4)

    cnv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(pool2)

    cnv5 = Activation('relu')(cnv5)
    cnv6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(cnv5)

    cnv6 = Activation('relu')(cnv6)
    cnv7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(cnv6)

    cnv7 = Activation('relu')(cnv7)
    pool3 = MaxPooling2D((2, 2),name='pool3_avg')(cnv7)
    # shape = (256, nb_rows/8, nb_cols/8)

    cnv8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(pool3)

    cnv8 = Activation('relu')(cnv8)
    cnv9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(cnv8)

    cnv9= Activation('relu')(cnv9)
    cnv10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(cnv9)

    cnv10 = Activation('relu')(cnv10)
    pool4 = MaxPooling2D((2, 2),name='pool4_avg')(cnv10)
    # shape = (512, nb_rows/16, nb_cols/16)

    cnv11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(pool4)

    cnv11 = Activation('relu')(cnv11)
    cnv12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(cnv11)

    cnv12 = Activation('relu')(cnv12)
    cnv13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')(cnv12)

    cnv13 = Activation('relu')(cnv13)
    pool5 = MaxPooling2D((2, 2),name='pool5_avg')(cnv13)
    # shape = (512, nb_rows/32, nb_cols/32)

    cnv14 = Conv2D(filters=4096, kernel_size=(7, 7), padding='same')(pool5)
    cnv14 = Activation('relu')(cnv14)
    cnv14 = Dropout(0.5)(cnv14)
    
    
    cnv15 = Conv2D(filters=4096, kernel_size=(1, 1), padding='same')(cnv14)
    cnv15 = Activation('relu')(cnv15)
    cnv15 = Dropout(0.5)(cnv15)
    
    # Encoder is built
    dcnv1 = Conv2D(filters=class_num, kernel_size=(1, 1), padding='same')(cnv15)
    unpool1 = Conv2DTranspose(filters=class_num, kernel_size=(3, 3),strides=(2,2), padding='same')(dcnv1)


    dcnv3 = Conv2D(filters=class_num, kernel_size=(1, 1), padding='same')(pool4)
    merge1=add([unpool1,dcnv3]) 
    dcnv4 = Conv2DTranspose(filters=class_num, kernel_size=(4, 4),strides=(2,2))(merge1)
    dcnv4 = Cropping2D(cropping=(1,1))(dcnv4)#crop theano ConvT output shape


    dcnv5=Conv2D(filters=class_num, kernel_size=(4, 4), padding='same')(pool3)
    merge2=add([dcnv4,dcnv5])
    dcnv6 = Conv2DTranspose(filters=class_num, kernel_size=(8, 8),strides=(8,8))(merge2)
#    dcnv6 = Cropping2D(cropping=(1,1))(dcnv6)#crop theano ConvT output shape


    output=Conv2D(filters=class_num, kernel_size=(4, 4), padding='same')(dcnv6)
    
    out=Lambda(lambda x:x+0., name='output')(output)
    out_2=Lambda(lambda x:x+0., name='output_2')(output)

    model = Model(inputs=input_node, outputs=[out,out_2])

    return model
#    
