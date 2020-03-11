import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from code import darknet
from code.yolov3 import Yolo3

def model(self, input_data, is_training):
    _, _, input_data = darknet.backbone(input_data, is_training)  # 1,13,13,1024
    # 全局平均池化
    input_data = tf.keras.layers.GlobalAveragePooling2D().apply(input_data)
    # input_data = tf.layers.average_pooling2d(inputs=input_data,pool_size=13,strides=13,padding='same')
    input_data = tf.layers.dense(inputs=input_data, units=399, activation=None,
                                 bias_initializer=tf.zeros_initializer())
    input_data = tf.reshape(input_data, (-1, 399))
    return input_data

def get_variables():
    x_input = tf.placeholder(tf.float32,shape=[None,224,224,3],name='x_input')
    y_input = tf.placeholder(tf.float32,shape=[None,399])

    ses = tf.Session()
    save = tf.train.Saver(ses,)
