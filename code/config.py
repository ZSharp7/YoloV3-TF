# _*_coding:utf-8_*_

from easydict import EasyDict as edict
import numpy as  np
import os
np.random.seed(1)
__M =edict()
cfg = __M


#================== YOLO ====================#

__M.Main =edict()

__M.Main.classes = './data/classes/classes.names'
__M.Main.anchors = './data/anchors/anchors2.txt'
__M.Main.strides = [8,16,32]
__M.Main.backbone = 0# 0:darknet53, 1:mobilenet_v1

#=================== Train ===================#

__M.Train =edict()

# 训练数据地址
__M.Train.xml = './data/images&xml/xml'
__M.Train.label = './data/label/train.txt'
# __M.Train.input_size = np.random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
# input图像大小
__M.Train.input_size = np.random.choice([416])
# 训练参数: 轮次, 批次, train样本数
__M.Train.epoch = 10
__M.Train.batch_size = 6
__M.Train.sample_num = len(open(__M.Train.label,'r').readlines())
# 数据增强(来自github yangyuan)
__M.Train.data_aug = True
__M.Train.box_maxnum = 100 # 回收盒大小
# 模型保存地址(用于restore和save)
__M.Train.darknet_savefile = './checkpoint/darknet_yolo3/'
__M.Train.mobilenet_savefile = './checkpoint/mobilenet_yolo3/'
# 两步式训练
__M.Train.is_twostep=False
__M.Train.one_step = 10
__M.Train.two_step = 100
__M.Train.is_training = True

#=================== Test ===================#

__M.Test =edict()

__M.Test.xml = './data/images&xml/xml'
__M.Test.label = './data/label/test.txt'
__M.Test.data_aug = False
__M.Test.is_training = False
__M.Test.model_savefile = './checkpoint/darknet_yolo3/'
# nms 阈值
__M.Test.nms_score = 0.45
# bbox 评分阈值
__M.Test.bboxes_score = 0.5

#=================== export ==================#
__M.Export = edict()
__M.Export.version=1
__M.Export.name = 'yolo3-darknet'