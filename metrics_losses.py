import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.losses import binary_crossentropy , BinaryCrossentropy
import tensorflow.keras.backend as k
eps = k.epsilon()


def cast_flat(true, pred):
    true_f = k.cast(true, 'float32')
    pred_f = k.cast(pred, 'float32')
    true = k.flatten(true_f)
    pred = k.flatten(pred_f)

    return true, pred

def iou(true, pred):
    eps = k.epsilon()
    true, pred = cast_flat(true, pred)
    intersection = k.sum(true * pred)
    union = (k.sum(true) + k.sum(pred)) - intersection
    iou = (intersection + eps) / (union + eps)

    return iou

def iou_loss(true, pred):
    return 1 - iou(true, pred)

def dice(true, pred):
    eps = k.epsilon()
    true, pred = cast_flat(true, pred)
    intersection = k.sum(true * pred)
    union = (k.sum(true) + k.sum(pred)) 
    dice = (2 * intersection + eps) / (union + eps)

    return dice

def dice_loss (true, pred):
    return 1 - dice(true, pred)

def dice_xent(true, pred):
    return dice_loss(true, pred) + binary_crossentropy(true, pred)

###################

def conf_matrix (true, pred, iou_threshold = 0.5):
  iou_score = iou(true, pred)
  TP, FP, FN = 0, 0, 0

  pos_class = 1. if k.sum (true) > 0  else 0.

  yt, yp = cast_flat(true, pred)

  if k.sum(true) > 0 and k.sum(pred) > 0:
    if iou_score >= iou_threshold:
      TP += 1
    else:
      FP += 1

  elif k.sum(true) > 0 and k.sum(pred) <= 0:
    FN += 1

  elif k.sum(true) <= 0 and k.sum(pred) > 0:
    FP += 1

  TP, FP, pos_class = (k.cast(count, 'float32') for count in (TP, FP, pos_class) )
  return TP, FP, pos_class

#####################################################

def precsn(true, pred, iou_threshold = 0.5):
  eps = k.epsilon()
  TP, FP, pos_class = conf_matrix (true, pred, iou_threshold)
  return (TP + eps ) / (TP + FP + eps)

def recall(true, pred, iou_threshold = 0.5):
  eps = k.epsilon()
  TP, FP, pos_class = conf_matrix (true, pred, iou_threshold)
  return (TP + eps) / (pos_class + eps )


def mAP (true, pred):
  prcsn_scores = []
  for iou_thresold in np.arange(.5,.95,0.05):
    prcsn_scores.append(precsn(true, pred, iou_thresold))

  mAP = np.sum(prcsn_scores)/len(prcsn_scores)
  return mAP

