import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

def bottleneck(inputs,channels,t,strides,name,is_training):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(name):
        output = tc.layers.conv2d(inputs=inputs,num_outputs=t*shape[-1],kernel_size=1,activation_fn=tf.nn.relu6,normalizer_fn=tc.layers.batch_norm,normalizer_params={'is_training': is_training})
        output = tc.layers.separable_conv2d(inputs=output,num_outputs=None,kernel_size=3,depth_multiplier=1,stride=strides,activation_fn=tf.nn.relu6,normalizer_fn=tc.layers.batch_norm,normalizer_params={'is_training': is_training})
        output = tc.layers.conv2d(inputs=output,num_outputs=channels,kernel_size=1,activation_fn=None,normalizer_fn=tc.layers.batch_norm,normalizer_params={'is_training': is_training})
        if shape[-1] == channels:
            output = tf.add(inputs,output)
    return output
def backbone(input_value,is_training):
    # mobilenet v2
    with tf.variable_scope ('MobileNet_V1'):
        with tf.variable_scope('V2Net_1_conv2d'):
            output = tc.layers.conv2d(inputs=input_value,num_outputs=32,kernel_size=3,stride=2,normalizer_fn=tc.layers.batch_norm,normalizer_params={'is_training': is_training})

        output = bottleneck(inputs=output,channels=16,t=1,strides=1,is_training=is_training,name='V2Net_2_bottleneck')

        output = bottleneck(inputs=output,channels=24,t=6,strides=2,is_training=is_training,name='V2Net_3_bottleneck')
        output = bottleneck(inputs=output,channels=24,t=6,strides=1,is_training=is_training,name='V2Net_4_bottleneck')

        output = bottleneck(inputs=output,channels=32,t=6,strides=2,is_training=is_training,name='V2Net_5_bottleneck')
        output = bottleneck(inputs=output,channels=32,t=6,strides=1,is_training=is_training,name='V2Net_6_bottleneck')
        output = bottleneck(inputs=output,channels=32,t=6,strides=1,is_training=is_training,name='V2Net_7_bottleneck')
        # print(output.shape,'52')
        route1 = output
        output = bottleneck(inputs=output,channels=64,t=6,strides=2,is_training=is_training,name='V2Net_8_bottleneck')
        output = bottleneck(inputs=output,channels=64,t=6,strides=1,is_training=is_training,name='V2Net_9_bottleneck')
        output = bottleneck(inputs=output,channels=64,t=6,strides=1,is_training=is_training,name='V2Net_10_bottleneck')
        output = bottleneck(inputs=output,channels=64,t=6,strides=1,is_training=is_training,name='V2Net_11_bottleneck')
        # print(output.shape,'26')
        route2 = output
        output = bottleneck(inputs=output,channels=96,t=6,strides=1,is_training=is_training,name='V2Net_12_bottleneck')
        output = bottleneck(inputs=output,channels=96,t=6,strides=1,is_training=is_training,name='V2Net_13_bottleneck')
        output = bottleneck(inputs=output,channels=96,t=6,strides=1,is_training=is_training,name='V2Net_14_bottleneck')

        output = bottleneck(inputs=output,channels=160,t=6,strides=2,is_training=is_training,name='V2Net_15_bottleneck')
        output = bottleneck(inputs=output,channels=160,t=6,strides=1,is_training=is_training,name='V2Net_16_bottleneck')
        output = bottleneck(inputs=output,channels=160,t=6,strides=1,is_training=is_training,name='V2Net_17_bottleneck')

        output = bottleneck(inputs=output,channels=320,t=6,strides=1,is_training=is_training,name='V2Net_18_bottleneck')
    # print(output.shape)

    return route1,route2,output