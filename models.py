from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy as BCE
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint
from  tensorflow.keras.optimizers import Adam
from metrics_losses import dice_xent, iou, precsn, recall, mAP
%load_ext tensorboard
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

class train:

    def __init__(self, model, log_path, checkpnt_path, lr = 0.001, round = 0) -> None:
        self.earlyStop = EarlyStopping(monitor='val_loss', min_delta= 0.001, patience= 8, verbose= 1, restore_best_weights= True)
        self.tensorboard = TensorBoard(log_dir= log_path)
        self.csvlogger = CSVLogger(log_path + f'training_{round}.log')
        self.checkpointer = ModelCheckpoint(checkpnt_path + "mchp_{round}_{epoch:04d}.hdf5", verbose= 1, save_weights_only= True )
        self.loss_dct = {
            "U_MASKS": dice_xent,
            "U_BORDERS": dice_xent}

        self.loss_weights = {"U_MASKS": 1.0, "U_BORDERS": 1.0}
        self.optimizer = Adam(learning_rate= lr)
        self.model = model.compile(optimizer= self.optimizer, loss = self.loss_dct, loss_weights= self.loss_weights , 
          metrics=  [iou, mAP, precsn, recall])

        
    def fit(self, epochs, train_gen, val_gen, trainsteps, valsteps, batchsize):

        callbacks = [self.csvlogger, self.checkpointer, self.tensorboard]

        results = self.model.fit( x = train_gen, validation_data = val_gen,
                            verbose= 1, 
                            steps_per_epoch= trainsteps, 
                            validation_steps= valsteps, 
                            batch_size= batchsize, 
                            callbacks = callbacks , 
                            epochs = epochs)

        return results

    def evaluate (model_weights, test_gen):
        