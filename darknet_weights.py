import tensorflow as tf
import numpy as np
from code.yolov3 import Yolo3
from code import config
cfg = config.cfg
import os
import time
import darknet53_train_ckpt_dataset
file_name = './data/yolov3.weights'
from code import darknet

def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """

    with open(weights_file, "rb") as fp:
        # This is verry import for count,it include the version of yolo
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []

    # os._exit(0)
    while i < len(var_list) - 1:
        # detector/darknet-53/Conv/BatchNorm/
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        # print('var1:',var1.name.split('/')[-2],'+',var1.name)
        if 'darknet53' not in var1.name.split('/')[-3] and 'darknet53' not in var1.name.split('/')[-4]:
            break
        if 'conv' in var1.name.split('/')[-2]:
            # check type of next layer,BatchNorm param first of weight
            # print('var2:',var1.name.split('/')[-2],'+',var2.name)
            if 'batch_norm' in var2.name.split('/')[-2]:
                # load batch norm params, It's equal to l.biases,l.scales,l.rolling_mean,l.rolling_variance

                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [gamma, beta, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    print(var, ptr)
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 5
            elif 'conv' in var2.name.split('/')[-2]:
                # load biases,not use the batch norm,So just only load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                print(bias, ptr)
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1

                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape(shape[3], shape[2], shape[0], shape[1])
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                print(var1, ptr)
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            # i+=1



        else:
            os._exit(0)
    return assign_ops


if __name__ == '__main__':

    print('darknet weight file is loading....')
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3],'x_input')
    print('darknet_weight is loading..')
    # with tf.variable_scope('model'):
    _,_,model = darknet.backbone(inputs,True)

    print('weight file is open, loading...')
    model_vars = tf.global_variables(scope='darknet53')

    load_ops = load_weights(model_vars, './data/darknet53.weights')
    print('create train.saver , loading...')
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(load_ops)
    # 将权重保存为ckpt文件

    print('start save file, loading ...')
    saver.save(sess, "./checkpoint/darknet/darknet_weight.ckpt")
    print('is done!')
