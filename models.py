from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy as BCE
from tensorflow.keras.metrics import MeanIoU, BinaryAccuracy, Accuracy
import h5py
import pandas as pd
import numpy as np



class modelBuilder:

    
    def __init__(self, input_shape) -> None:
        self.input_shape = input_shape
        self.input_layer = Input (input_shape)

    def __conv_unit(self, inputs, nfilters):
        x = Conv2D(nfilters, kernel_size= 3, padding= 'same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)

        x = Conv2D(nfilters, 3, padding= 'same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        return x

    def _encoder_unit(self, inputs, nfilters):
        x = self.__conv_unit(inputs, nfilters)
        p = MaxPool2D()(x)
        return x, p

    def _decoder_unit(self, inputs, skpdfeatures, nfilters):
        x = Conv2DTranspose(nfilters, kernel_size= 2, strides = 2, padding = 'same')(inputs)
        x = Concatenate()([x, skpdfeatures])
        x = self.__conv_unit(x, nfilters)
        return x

    def _out_unit (self, out_list, branch_name):
        out = self.__conv_unit(out_list, 64)
        out = self.__conv_unit(out, 32)
        out = Conv2D(1, 1, padding= 'same', activation= 'sigmoid', name = branch_name)(out)

        return out



    def simple_unet (self, name = 'U-SIMPLE'):

        # ENCODE #
        x1, p1 = self._encoder_unit(self.input_layer, 32)
        x2, p2 = self._encoder_unit(p1, 64)
        x3, p3 = self._encoder_unit(p2, 128)
        x4, p4 = self._encoder_unit(p3, 256)

        # BRIDGE #
        b = self.__conv_unit(p4, 512)

        # DECODE #
        y1 = self._decoder_unit (b, x4, 256 )
        y2 = self._decoder_unit (y1, x3, 128)
        y3 = self._decoder_unit (y2, x2, 64)
        y4 = self._decoder_unit (y3, x1, 32)

        # OUTPUT #
        output = Conv2D(1, 1, padding= 'same', activation= 'sigmoid')(y4)

        model = Model(inputs = self.input_layer, outputs = output, name = name)

        return model

    def compound_unet(self):
        
        UNET_masks = self.simple_unet(name = 'U_MASKS')
        UNET_borders = self.simple_unet( name = 'U_BORDERS')
        output_list = Concatenate()([UNET_masks.output, UNET_borders.output])

        # Encode Outs
        out1 = self._out_unit(output_list, 'U_MASKS')
        out2 = self._out_unit(output_list, 'U_BORDERS')

        model = Model(inputs = self.input_layer, outputs = [out1, out2], name = 'UNET_ARCH')

        return  model
