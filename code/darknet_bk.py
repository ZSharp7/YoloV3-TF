# _*_coding:utf-8_*_
import tensorflow as tf
from code import config
cfg = config.cfg

# 1. convolutional 层
# con2d layer >>  BN layer >> LeakyReluLayer

# 2. Residual层
# 1*1 Convolutional >> 3*3 Convolutional

# 3. 构建darknet-53

def fill_padding(input_value, kernel):
    '''
    构建darknet53中的下采样操作
    :param input_value: 被padding 的张量
    :param kernel: 卷积核大小
    :return:
    附: 填充三通道图片, 维度: [N,W,H,C]
    '''
    # fill = kernel - 2
    # fill_b = fill // 2 + 1
    # fill_e = fill // 2 + 1
    # # fill = tf.concat([[0, 0], [fill_b, fill_e], [fill_b, fill_e], [0, 0]])
    # fill_input = tf.pad(input_value, [[0, 0], [fill_b, fill_e], [fill_b, fill_e], [0, 0]])
    pad_h, pad_w = (kernel - 2) // 2 + 1, (kernel - 2) // 2 + 1
    paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    input_data = tf.pad(input_value, paddings, 'CONSTANT')
    return input_data


def convolutional_layer(input_value, filter, kernel_size, name, is_downsample=False, is_bn=True, is_active=True,is_training=True):
    '''
    convolutional层
    :param input_value: 输入张量
    :param kernel: 卷积核大小
    :param name: 空间命名
    :param is_upsample: 是否下采样
    :param is_bn: 是否bn操作
    :param is_active: 是否激活(leaky_relu,alpha=0.11)
    :return:
    '''

    with tf.name_scope(name):
        if is_downsample:
            input_value = fill_padding(input_value, kernel_size)
            padding = 'VALID'
            strides = 2
        else:
            padding = 'SAME'
            strides = 1
        input_value = tf.layers.conv2d(inputs=input_value, filters=filter,
                                       kernel_size=kernel_size, strides=strides,trainable=is_training,
                                       padding=padding)
        if is_bn:
            input_value = tf.layers.batch_normalization(input_value, training=is_training)

        if is_active:
            input_value = tf.nn.leaky_relu(input_value, alpha=0.11)
            # input_value = tf.nn.relu6(input_value)
    return input_value


def residual_layer(input_value, filter, name,is_training):
    residual = input_value
    with tf.variable_scope(name):
        input_value = convolutional_layer(input_value=input_value, filter=filter, kernel_size=1, name='residual_conv1',is_training=is_training)
        input_value = convolutional_layer(input_value=input_value, filter=2 * filter, kernel_size=3,
                                          name='residual_conv2',is_training=is_training)

    input_value += residual
    return input_value


def backbone(input_value,is_training):
    # darknet53
    '''
    构建darknet53网络前52层
    :param input_value:
    :return:
    '''
    with tf.variable_scope('darknet53'):
        input_value = convolutional_layer(input_value=input_value, filter=32, kernel_size=3, name='conv1',is_training=is_training)
        input_value = convolutional_layer(input_value=input_value, filter=64, kernel_size=3, is_downsample=True, name='conv2',is_training=is_training)

        input_value = residual_layer(input_value, filter=32, name='residual_1',is_training=is_training)
        input_value = convolutional_layer(input_value=input_value, filter=128, kernel_size=3, is_downsample=True, name='conv4',is_training=is_training)

        for i in range(2):
            input_value = residual_layer(input_value, filter=64, name='residual_%d' % (i + 2),is_training=is_training)
        input_value = convolutional_layer(input_value=input_value, filter=256, kernel_size=3, is_downsample=True, name='conv5',is_training=is_training)

        for i in range(8):
            input_value = residual_layer(input_value, filter=128, name='residual_%d' % (i + 4),is_training=is_training)
        route1 = input_value
        input_value = convolutional_layer(input_value=input_value, filter=512, kernel_size=3, is_downsample=True, name='conv6',is_training=is_training)


        for i in range(8):
            input_value = residual_layer(input_value, filter=256, name='residual_%d' % (i + 12),is_training=is_training)
        route2 = input_value
        input_value = convolutional_layer(input_value=input_value, filter=1024, kernel_size=3, is_downsample=True, name='conv7',is_training=is_training)


        for i in range(4):
            input_value = residual_layer(input_value, filter=512, name='residual_%d' % (i + 20),is_training=is_training)

    return route1, route2, input_value


