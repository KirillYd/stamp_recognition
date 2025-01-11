import os
import math
import string

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import get_value, ctc_decode
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam

alphabet = ' !"%()+,-./0123456789:;<=>?@[]~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ№'

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred, input_length, greedy=True)

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))
        y_true_sparse = tf.sparse.retain(y_true_sparse, tf.not_equal(y_true_sparse.values, tf.math.reduce_max(y_true_sparse.values)))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(K.cast(tf.shape(y_true)[0], 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


def CTCLoss(y_true, y_pred):
    """
    Compute the training-time loss value
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")


    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def crnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01),
                     input_shape=(256, 32, 1)))  # tf.keras.layers.LeakyReLU
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2)))
    model.add(BatchNormalization())  # x 1 x ..

    model.add(Reshape((-1, 256)))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))

    model.add(Dense(len(alphabet) + 1, activation='softmax'))  # +1 for ctc blank

    model.summary()

    model.compile(optimizer=Nadam(learning_rate=0.001, clipnorm=1.0), loss=CTCLoss, metrics=[CERMetric()])
  #
    return model



# Decode label for single image

def num_to_label(num, alphabet):
    text = ""
    for ch in num:
        if ch == len(alphabet): # ctc blank
          break
        else:
          text += alphabet[ch]
    return text


# Decode labels for softmax matrix

def decode_text(nums):
    values = get_value(
        ctc_decode(nums, input_length=np.ones(nums.shape[0])*nums.shape[1], greedy=True)[0][0])

    texts = []
    for i in range(nums.shape[0]):
        value = values[i]
        texts.append(num_to_label(value[value >= 0], alphabet))
    return texts

def preprocess(img):
    for func in [resize_n_rotate]:
        img = func(img)
    return img.astype("float32")/255


def resize_n_rotate(img, shape_to=(32, 256)): #shepe_to 64, 800
    if img.shape[0] > shape_to[0] or img.shape[1] > shape_to[1]:
        shrink_multiplayer = min(math.floor(shape_to[0] / img.shape[0] * 100) / 100,
                                 math.floor(shape_to[1] / img.shape[1] * 100) / 100)
        img = cv2.resize(img, None,
                         fx=shrink_multiplayer,
                         fy=shrink_multiplayer,
                         interpolation=cv2.INTER_AREA)

    img = cv2.copyMakeBorder(img, math.ceil(shape_to[0]/2) - math.ceil(img.shape[0]/2),
                           math.floor(shape_to[0]/2) - math.floor(img.shape[0]/2),
                           0, #math.ceil(shape_to[1]/2) - math.ceil(img.shape[1]/2)
                           math.floor(shape_to[1]) - math.floor(img.shape[1]),
                           cv2.BORDER_CONSTANT, value=255)

    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def predict_text(model, img):
    # predicts = decode_text(pred_img)



    processed_img = preprocess(img)
    input_data = np.array([processed_img])

    pred_img = model.predict(input_data)
    predictions = decode_text(pred_img)
    return predictions