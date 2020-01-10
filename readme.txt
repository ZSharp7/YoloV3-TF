YOLO3-tensorflow
# 目录结构
    - checkpoint: 检查点
        - darknet:              darknet weight解析后, ckpt的存放地址
        - darknet_yolo3:        模型训练保存地址
        - mobilenet:            mobilenet weight解析后, ckpt的存放地址(暂无)
        - mobilenet_yolo3:      模型训练保存地址

    - code: 模型代码
        - __init__.py
        - config.py:            模型配置文件
        - darknet.py:           darknet网络搭建
        - data_pre.py:          数据处理文件,(iter)
        - kmeans.py:            无监督聚类, anchors(暂未实现)
        - mobilenet.py:         mobilenet网络搭建(V2)
        - yolov3.py:            yolov3网络搭建
        - dataset_TFRecord.py   创建TFRecord格式的tf.data.Dataset数据集

    - data: 模型数据
        - anchors:              anchors文件存放地址
        - classes:              cls label 存放地址
        - fonts:                用于test时, 字体显示
        - images&xml:           图片数据和xml数据存放地址
        - label:                解析xml后, 数据存放地址
        - test:                 测试图片
        - 文件yolov3.weights    yolov3 的backbone权重数据
        - 文件yolov3_1.weights  备份

    - log: summary

    - 文件darknet_weight_tockpt.py  解析darknet的weight文件, 并保存为ckpt
    - 文件labelimg_yolov3.py        解析xml, 生成训练和测试数据
    - 文件test.py                   测试, 以图片形式显示(未保存)
    - 文件train.py                  模型训练
    - 文件train_TFRecord.py         使用TFRecord数据集进行训练
    - 文件readme.txt                项目描述

