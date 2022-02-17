import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy as BCE
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler
from  tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from metrics_losses import dice_xent, iou
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import cv2 as cv
from metrics_losses import metrics 
from preprocess import single_overlay
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects

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

    def __init__(self, simple = False, log_path = '', checkpnt_path = '', lr = 0.001, input_shape = (256, 256, 3)) -> None:
        super().__init__(input_shape)

        if simple:
            self.model = self.simple_unet()
            self.loss = dice_xent 
            self.loss_weights = None
            self.monitor = 'iou'
        else:
            self.model = self.compound_unet()
            self.loss = { "U_MASKS": dice_xent, "U_BORDERS": dice_xent }
            self.loss_weights = {"U_MASKS": 1.0, "U_BORDERS": 1.0}
            self.monitor = 'U_MASKS_iou'

        self.lr = lr
        self.lr_downstep = 20
        self.decay = 0.75
        self.optimizer = Adam(learning_rate= self.lr)

        self.summary = self.model.summary
        self.plot = plot_model(self.model, show_shapes=True)                                                           
        self.log_path = log_path
        self.checkpnt_path = checkpnt_path

        metrcs = metrics()
        self.model.compile(
        optimizer= self.optimizer, 
        loss= self.loss,
        loss_weights = self.loss_weights,
        metrics= [iou, metrcs.mAP, metrcs.precsn, metrcs.recall])
     

    def schedule(self, epoch):
        return self.lr * (self.decay ** np.floor(epoch/self.lr_downstep))


    def fit(self, train_gen, val_gen, round = 0, length = (3342, 836, 1045 ), user_dct = None): 
        len_train, len_val, len_test = length
        self.len_test = len_test
        self.round = round
        self.epochs = 100
        self.batchsize = 40
        self.trainsteps = len_train // self.batchsize
        self.valsteps = len_val // self.batchsize
        self.teststeps = len_test

        if user_dct:
            for k,v in user_dct.items():
                if k in self.__dict__:
                    setattr(self, k, v)
                else:
                    raise KeyError(k)

        self.earlyStop = EarlyStopping(monitor= self.monitor, min_delta= 0.001, patience= 10, verbose= 1, restore_best_weights= True)
        self.tensorboard = TensorBoard(log_dir= self.log_path)
        self.csvlogger = CSVLogger(self.log_path + f'training_{self.round}.log')
        # self.checkpointer = ModelCheckpoint(self.checkpnt_path + f'{self.round}_' +"mchp_{epoch:04d}.hdf5", verbose= 1, save_weights_only= True )  
        self.checkpointer = ModelCheckpoint(self.checkpnt_path + f'{self.round}_' + "best_model.hdf5", monitor = self.monitor, mode = 'max', verbose= 1, save_best_only= True, save_weights_only= True )  
        self.lr_sched = LearningRateScheduler (self.schedule, verbose= 1)
        callbacks = [self.lr_sched, self.csvlogger, self.checkpointer, self.tensorboard]

        results = self.model.fit( x = train_gen, validation_data = val_gen,
                            verbose= 1, 
                            steps_per_epoch= self.trainsteps, 
                            validation_steps= self.valsteps, 
                            batch_size= self.batchsize, 
                            callbacks = callbacks , 
                            epochs = self.epochs)

        return results

    pix_sup= np.vectorize(lambda x: 0 if x < 0.5 else 1)

    def evaluate(self, test_gen):
        test_scores = self.model.evaluate(test_gen, batch_size= 96 , steps= self.len_test // 96,  return_dict= True)
        return test_scores

    def load_weights(self, weightpath):
        model = self.model
        model.load_weights(weightpath)
        self.model = model

    def prec_recall(self, test_gen, neg_data = False):
        iouthresh_steps =[i for i in np.arange (0.1,1,.05)]
        p_curve = []
        r_curve = []

        for step in iouthresh_steps:
            met = metrics(step)
            metrics_list = [iou, met.mAP, met.precsn, met.recall]
            self.model.compile(
                        optimizer= self.optimizer, 
                        loss= self.loss,
                        loss_weights = self.loss_weights,
                        metrics= metrics_list)
            test_scores = self.model.evaluate(test_gen, batch_size= 96 , steps= self.len_test // 96,  return_dict= True)
            precision =  test_scores['precsn']
            recall =  test_scores['recall']
            p_curve.append (precision)
            r_curve.append (recall) if neg_data else r_curve.append(0) 
        

        self.plot_presRecall(iouthresh_steps, p_curve, r_curve)

        return p_curve, r_curve

    @staticmethod
    def plot_presRecall(steps, p_curve, r_curve):

        x = [np.round(i, 2) for i in np.arange (0.1, 1, .1)]
        width = 0.05

        plt.figure(figsize = (20,8))
        plt.subplot(121)
        plt.bar(x, width= width, height= p_curve[::2])
        plt.xticks(x, x)
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.xlabel('IoU Threshold', fontsize = 14)
        plt.ylabel ('Average Precision', fontsize = 14 )
        plt.suptitle('Evaluation', y = 1.02, fontsize = 18)

        plt.subplot(122)
        plt.plot(steps, p_curve)
        if sum(r_curve) > 0 : plt.plot(steps, r_curve)
        plt.xlabel('IoU Threshold', fontsize = 14)
        plt.xticks(steps, rotation = 20)
        plt.yticks(np.arange(0,1.05, 0.05))
        plt.ylabel('Precision | Recall', fontsize = 14)
        plt.legend(['Precision', 'Recall'], loc='best')

        plt.tight_layout()
        plt.show()



    @staticmethod
    def label_overlay(ximg, mask, border, test = False):
        
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

    def make_pred(self, xt):

        p_label = self.model.predict(xt)

        if len(p_label) == 2:
            p_mask, p_border = p_label  
        else:
            p_mask, p_border  = p_label, np.zeros_like(p_label)

        return p_mask, p_border

    @staticmethod
    def process_pred(p_mask, min_size = 1000):
        pixel_clip = np.vectorize(lambda pixel: 0 if pixel < 0.5 else 1)
        p_mask = pixel_clip(p_mask.squeeze())
        labels = label(p_mask, background = 0)
        labels = remove_small_objects(labels, min_size = min_size)
        props = [p for p in regionprops(labels)]
        areas = [p.area for p in props] 

        bboxs = []
        if len(areas) > 0:
            sorted_inds = sorted (np.argsort(areas), reverse = 1)
            for ind in sorted_inds:
                max_area = np.argmax(areas)
                candidate = props[ind]
                bbox_coords = candidate.bbox
                bboxs.append(bbox_coords)
        
        return bboxs, labels

    def inspect_preds(self, testgen, length):
        ovs = []
        for i in range(length):
            X ,_ = next(testgen)
            p_mask, p_border = self.make_pred(X)
            bboxs, labels = self.process_pred(p_mask, min_size = 1000)
            _ , border_labels = self.process_pred(p_border, min_size = 200)
            ov = self.label_overlay(X, labels, border_labels, test = True)

            if len(bboxs) > 0:
                for bbx in bboxs:
                    minr, minc, maxr, maxc = bbx
                    start, end = (minc, minr) , (maxc, maxr)
                    ov = cv.rectangle(ov, start, end, color = (36,255,12 ), thickness = 2)
                ovs.append(ov)
        return ovs

class loadModel(fitEval):

    def __init__(self, simple=True, lr= 0.00001, input_shape=(256, 256, 3)) -> None:
        super().__init__(simple,lr, input_shape)
        self.weightpath = ''
        
    def load_model(self, weightpath):
        self.weightpath = weightpath
        return self.load_weights(self.weightpath)

        


