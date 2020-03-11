YOLO3-tensorflow
# 目录结构
    - checkpoint: 检查点
        - darknet_yolo3:         	模型训练保存地址
        - mobilenet_yolo3:      	模型训练保存地址

    - code: 模型代码
        - __init__.py
        - config.py:             	模型配置文件
        - darknet.py:           	darknet网络搭建
        - data_pre.py:          	数据处理文件,(iter)
        - mobilenet.py:        	mobilenet网络搭建(V2)
        - yolov3.py:              	yolov3网络搭建


    - data: 模型数据
        - anchors:              	anchors文件存放地址
        - classes:              	cls label 存放地址
        - fonts:               	用于test时, 字体显示
        - images&xml:     	图片数据和xml数据存放地址
        - label:           	     	解析xml后, 数据存放地址
        - test:                	测试图片


    - log: summary

    - 文件darknet_weight_tockpt.py  	解析darknet的weight文件, 并保存为ckpt
    - 文件labelimg_yolov3.py        	解析xml, 生成训练和测试数据
    - 文件test.py                  		测试, 以图片形式显示(未保存)
    - 文件train_nograds.py                  	模型训练(未梯度累积)
    - 文件readme.txt                		项目描述
    - 文件darknet_to_yolo.py		生成darknet53训练数据
    - 文件darknet53_train_ckpt_dataset.py	darknet53网络训练文件(分段训练)
    - kmeans_run.py	kmeans		聚类bboxes



