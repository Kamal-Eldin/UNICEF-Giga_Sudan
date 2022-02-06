import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy as BCE
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint
from  tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from metrics_losses import dice_xent, iou, precsn, recall, mAP
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import cv2 as cv



class modelBuilder:

    
    def __init__(self, input_shape = (256, 256, 3)) -> None:
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



    def simple_unet (self, name = 'U_SIMPLE'):

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

class fitEval(modelBuilder):

    def __init__(self, simple = True, log_path = '', checkpnt_path = '', lr = 0.001, input_shape = (256, 256, 3)) -> None:
        super().__init__(input_shape)

        if simple:
            self.model = self.simple_unet()
            self.loss = dice_xent 
            self.loss_weights = None
        else:
            self.model = self.compound_unet()
            self.loss = { "U_MASKS": dice_xent, "U_BORDERS": dice_xent }
            self.loss_weights = {"U_MASKS": 1.0, "U_BORDERS": 1.0}

        self.optimizer = Adam(learning_rate= lr)
        self.summary = self.model.summary
        self.plot = plot_model(self.model, show_shapes=True)
                                                                          
        self.log_path = log_path
        self.checkpnt_path = checkpnt_path
        
        self.model.compile(
        optimizer= self.optimizer, 
        loss= self.loss,
        loss_weights = self.loss_weights,
        metrics= [iou, mAP, precsn, recall])
        
       

    def fit(self, train_gen, val_gen, trainsteps, valsteps, epochs = 3, batchsize = 1, round = 0): 
        self.round = round
        self.earlyStop = EarlyStopping(monitor='val_loss', min_delta= 0.001, patience= 8, verbose= 1, restore_best_weights= True)
        self.tensorboard = TensorBoard(log_dir= self.log_path)
        self.csvlogger = CSVLogger(self.log_path + f'training_{round}.log')
        self.checkpointer = ModelCheckpoint(self.checkpnt_path + f'{round}_' +"mchp_{epoch:04d}.hdf5", verbose= 1, save_weights_only= True )  
       
        callbacks = [self.csvlogger, self.checkpointer, self.tensorboard]

        results = self.model.fit( x = train_gen, validation_data = val_gen,
                            verbose= 1, 
                            steps_per_epoch= trainsteps, 
                            validation_steps= valsteps, 
                            batch_size= batchsize, 
                            callbacks = callbacks , 
                            epochs = epochs)

        return results


    def evaluate(self, test_gen, len_xtest):
        iouthresh_steps =[i for i in np.arange (0.1,1,.05)]
        p_curve = []
        r_curve = []

        for step in iouthresh_steps:
            p_scores = []
            r_scores = []
            for i in range (len_xtest):
                xt , [ymask, yborder] = next(test_gen)
                ypred = self.model.predict(xt)
                ypred = ypred[0] if type(ypred) is list else ypred
                p_scores.append (precsn(ymask, ypred, step))
                r_scores.append (recall(ymask, ypred, step))
                
            p_curve.append(np.mean(p_scores))
            r_curve.append(np.mean(r_scores))

        plt.figure(figsize = (12,8))
        plt.plot(iouthresh_steps, p_curve)
        plt.plot(iouthresh_steps, r_curve)
        plt.xlabel('IoU Threshold', fontsize = 14)
        plt.xticks(iouthresh_steps, rotation = 20)
        plt.yticks(np.arange(0,1.05, 0.05))
        plt.ylabel('Precision | Recall', fontsize = 14)
        plt.legend(['Precision', 'Recall'], loc='best')
        plt.show()

        return p_curve, r_curve

    pix_sup= np.vectorize(lambda x: 0 if x < 0.5 else 1)

    @staticmethod
    def imglabel_overlay(ximg, mask, border, test = False):
        
        pixel_clip = np.vectorize(lambda pixel: 0 if pixel < 0.5 else 1)

        mask_clr, brdr_clr = (255, 0, 0) , (0, 255, 0)
        if test: 
            mask_clr, brdr_clr = (0, 0, 255), (255, 255, 255) 
            mask, border  = (pixel_clip(label).squeeze() for label in (mask, border))

        image = np.copy(ximg)
        image /=  np.max(image)
        image *= 255
        image = image.astype(dtype = np.uint8).squeeze()

        mask *= 255
        border *= 255
        mask = mask.astype(dtype = np.uint8).squeeze()
        border = border.astype(dtype = np.uint8).squeeze()
        
        blue_canvas = np.full(image.shape, mask_clr, image.dtype)
        white_canvas = np.full(image.shape, brdr_clr, image.dtype)

        blueMask = cv.bitwise_and(blue_canvas, blue_canvas, mask=mask)
        whiteborder = cv.bitwise_and(white_canvas, white_canvas, mask=border)
        out = cv.addWeighted(blueMask, .5, image, 1, 0, image)
        out = cv.addWeighted(whiteborder, .5, out, 1, 0, out)
        return out

