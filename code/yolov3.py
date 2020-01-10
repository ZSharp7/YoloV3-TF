# _*_coding:utf-8_*_
import darknet
import tensorflow as tf
import data_pre
from config import cfg
import numpy as np
import mobilenet
# _anchors = [[10, 13], [16, 30], [33, 23],
#             [30, 61], [62, 45], [59, 119],
#             [116, 90], [156, 198], [373, 326]]
# _classes = 10


class Yolo3:
    def __init__(self, input_value,is_training):
        self.inputs = input_value
        self.anchors = data_pre.get_anchors(cfg.Main.anchors)
        self.classes = data_pre.get_classes(cfg.Main.classes)
        self.max_iou_ = 0.5
        self.strides = [8, 16, 32]
        self.backbone = cfg.Main.backbone
        self.is_training = is_training
        with tf.variable_scope('yolo3_model'):
            self.conv_one, self.conv_two, self.conv_three = self.__network()
            self.pre_one = self.__yolo_layer(self.conv_one, self.strides[0], self.anchors[6:9])
            self.pre_two = self.__yolo_layer(self.conv_two, self.strides[1], self.anchors[3:6])
            self.pre_three = self.__yolo_layer(self.conv_three, self.strides[2], self.anchors[3:6])

    def __network(self):
        '''
        构建网络, 输出conv盒子
        :return:
        '''

        if self.backbone == 0:
            route1, route2, inputs = darknet.backbone(input_value=self.inputs,is_training=self.is_training)
        else:
            route1, route2, inputs = mobilenet.backbone(input_value=self.inputs,is_training=self.is_training)
        conv_three, inputs = self.convolutional_set(inputs, 512, len(self.classes), 'conv_one')

        # inputs = self.upsample(inputs, name='upsample_one')
        inputs = tf.concat([inputs, route2], axis=-1)
        conv_two, inputs = self.convolutional_set(inputs, 256, len(self.classes), 'conv_two')


        # inputs = self.upsample(inputs, name='upsample_two')
        inputs = tf.concat([inputs, route1], axis=-1)
        conv_one, _ = self.convolutional_set(inputs, 128, len(self.classes), 'conv_three')

        return conv_one, conv_two, conv_three

    def __yolo_layer(self, input_value, strides, anchors):
        '''
        将预测值解码, 转换成框的中心点, 并返回
            预测结果对应着三个预测框的位置, reshape成(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 52, 52, 3, 85)
            N: 数据量
            13,13: 特征图大小
            3: anchors
            85: 4+1+80(
                    4: x_offset, y_offset, h, w
                    1: conf
                    80: cls
            )

        '''
        num_anchors = len(anchors)
        output_shape = tf.shape(input_value)

        # 输出维度
        output = tf.reshape(input_value,
                            (output_shape[0], output_shape[1], output_shape[1], num_anchors, 5 + len(self.classes)))

        # 分解维度
        # dxdy = output[:, :, :, :, 0:2]
        # dwdh = output[:, :, :, :, 2:4]
        # conf = output[:, :, :, :, 4:5]
        # cls = output[:, :, :, :, 5:]
        dxdy, dwdh, conf, cls = tf.split(output, [2, 2, 1, len(self.classes)], axis=-1)

        # 搭建g网格
        y = tf.tile(tf.range(output_shape[1], dtype=tf.int32)[:, tf.newaxis], [1, output_shape[1]])
        x = tf.tile(tf.range(output_shape[1], dtype=tf.int32)[tf.newaxis, :], [output_shape[1], 1])
        # x, y = tf.meshgrid(x, y)  # tf生成网格函数
        # x = tf.reshape(x,[-1,1])
        # y = tf.reshape(y,[-1,1])

        x_y =tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        x_y = tf.tile(x_y[tf.newaxis, :, :, tf.newaxis, :], [output_shape[0], 1, 1, 3, 1]) # 生成num_anchors个网格
        x_y = tf.cast(x_y,tf.float32)

        # 计算中心点
        b_xy = (x_y + tf.nn.sigmoid(dxdy))*strides
        b_wh = strides * (tf.exp(dwdh) * tf.to_float(anchors))
        # 置信度和分类
        b_conf = tf.nn.sigmoid(conf)
        b_cls = tf.nn.sigmoid(cls)

        # concat
        return tf.concat([b_xy, b_wh, b_conf, b_cls], axis=-1)

    def upsample(self, input_value, name):
        '''
        上采样, 使用最近邻插值
        '''
        with tf.variable_scope(name):
            input_shape = tf.shape(input_value)
            input_value = tf.image.resize_nearest_neighbor(input_value, size=[input_shape[1] * 2, input_shape[2] * 2])
        return input_value

    def convolutional_set(self, input_value, filter, n_classes, name):
        '''

        :param input_value:
        :param filter:
        :param n_anchors: 框的size
        :param n_classes: label的类别数量
        :param name:
        :return:
        '''
        with tf.variable_scope(name):
            input_value = darknet.convolutional_layer(input_value=input_value, filter=filter, kernel_size=1,
                                                      name='conv1')
            input_value = darknet.convolutional_layer(input_value=input_value, filter=filter * 2, kernel_size=3,
                                                      name='conv2')
            input_value = darknet.convolutional_layer(input_value=input_value, filter=filter, kernel_size=1,
                                                      name='conv3')
            input_value = darknet.convolutional_layer(input_value=input_value, filter=filter * 2, kernel_size=3,
                                                      name='conv4')
            input_value = darknet.convolutional_layer(input_value=input_value, filter=filter, kernel_size=1,
                                                      name='conv5')

            pre = darknet.convolutional_layer(input_value=input_value, filter=filter * 2, kernel_size=3,
                                              name='conv_pre1')
            pre = darknet.convolutional_layer(input_value=pre, filter=3 * (5 + n_classes), kernel_size=1,
                                              name='conv_pre2')

            input_value = darknet.convolutional_layer(input_value=input_value, filter=filter // 2, kernel_size=1,
                                                      name='conv6')
            inputs = self.upsample(input_value, name + '_upsample')

        return pre, inputs

    def focal(self, target, actual, alpha=1, gamma=2):
        '''
        求focal损失(针对类别不平衡问题) 置信度和bool
        https://www.cnblogs.com/xuanyuyt/p/7444468.html#_label3

        :param target: 置信度
        :param actual: bool是否为正样本
        :param alpha: 调制系数
        :param gamma: >=0
        :return:
        '''
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bboxes_giou_iou(self,boxes1,boxes2,type='iou'):
        # print(boxes1.shape,boxes2.shape)
        '''
        求iou和giou
        C \ (A ∪ B) 的面积为C的面积减去A∪B的面积。再用Ａ、Ｂ的IoU值减去这个比值得到GIoU。
        giou : https://giou.stanford.edu/GIoU.pdf
        :param boxes1:
        :param boxes2:
        :param type:
        :return:
        '''
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)

        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)

        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2]) # bbox最大左上角点集合
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:]) #bbox最小右下角点集合

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        if type == 'iou':
            return iou
        else:
            enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2]) # 求C的最小左上角点集合
            enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:]) # 求C的最大右下角点集合
            enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
            enclose_area = enclose[..., 0] * enclose[..., 1]
            giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

            return giou


    def loss_layer(self, conv, pre, bbox, recover, strides):
        '''
        构建损失, 输出边框损失, 置信度损失,分类损失
        构建流程:
            1. 标准化框值
            2. 计算iou去除小于阈值的框, 留下(m)框, 后面只对m框进行计算
            3.
        :param conv: conv三个
        :param pre: pre三个
        :param bbox: label框
        :param recover: true框(回收)
        :param strides:
        :return:
        '''
        batch_size = conv.shape[0]
        output_size = conv.shape[1]
        input_size = strides * output_size

        conv = tf.reshape(conv,(batch_size,output_size,output_size,3,5+len(self.classes)))
        # print(conv.shape,bbox.shape,pre.shape)
        conv_conf = conv[:,:,:,:,4:5] # 置信度
        conv_pro = conv[:,:,:,:,5:] # cls

        pre_xywh = pre[:,:,:,:,0:4] # xywh
        pre_conf = conv[:,:,:,:,4:5] #

        bbox_xywh = bbox[:,:,:,:,0:4] # xywh
        bbox_bool = bbox[:,:,:,:,4:5] # 0 或 1
        bbox_pro = bbox[:,:,:,:,5:] # cls

        # giou loss
        giou = tf.expand_dims (self.bboxes_giou_iou(pre_xywh,bbox_xywh,type='giou'),axis=-1)
        input_size = tf.cast(input_size,tf.float32)

        bbox_loss_scale = 2.0-1.0*bbox_xywh[:,:,:,:,2:3]*bbox_xywh[:,:,:,:,3:4] /(input_size*input_size) # input_size w和h
        giou_loss = bbox_bool * bbox_loss_scale * (1-giou)

        iou = self.bboxes_giou_iou(pre_xywh[:, :, :, :, np.newaxis, :], recover[:, np.newaxis, np.newaxis, np.newaxis, :, :],type='iou')
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1) # 使用iou进行筛选

        respond_bgd = (1.0 - bbox_bool) * tf.cast(max_iou < self.max_iou_, tf.float32)

        conf_focal = self.focal(bbox_bool, pre_conf)

        conf_loss = conf_focal * (
                bbox_bool * tf.nn.sigmoid_cross_entropy_with_logits(labels=bbox_bool, logits=conv_conf)+
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=bbox_bool, logits=conv_conf)
        )

        pro_loss = bbox_bool * tf.nn.sigmoid_cross_entropy_with_logits(labels=bbox_pro, logits=conv_pro)
        # print(giou_loss.shape,conf_loss.shape,pro_loss.shape)
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(pro_loss, axis=[1, 2, 3, 4]))

        return giou_loss,conf_loss,prob_loss

    def loss_total(self,one_bbox,two_bbox,three_bbox,one_recover,two_recover,three_recover):
        '''
        损失函数包含三个部分:
                    边界框损失
                    置信度损失
                    分类损失
        '''
        with tf.name_scope('small_box_loss'):
            loss_small = self.loss_layer(self.conv_one,self.pre_one,one_bbox,one_recover,self.strides[0])
        with tf.name_scope('medium_box_loss'):
            loss_medium = self.loss_layer(self.conv_two,self.pre_two,two_bbox,two_recover,self.strides[1])
        with tf.name_scope('big_box_loss'):
            loss_big = self.loss_layer(self.conv_three,self.pre_three,three_bbox,three_recover,self.strides[2])
        with tf.name_scope('giou_loss'):
            giou_loss = loss_small[0]+loss_medium[0]+loss_big[0]
        with tf.name_scope('conf_loss'):
            conf_loss = loss_small[1] + loss_medium[1] + loss_big[1]
        with tf.name_scope('pro_loss'):
            pro_loss = loss_small[2] + loss_medium[2] + loss_big[2]
        return giou_loss,conf_loss,pro_loss


