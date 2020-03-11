

# yolo3

## 一. 运行环境

1. tensorflow==1.13.1
2. numpy==1.16.2
3. python==3.7.3
4. opencv
5. easydict

## 二.使用方法

1. 数据准备

   将image放入'./data/images&xml/images'下
   将xml放入'./data/images&xml/xml'下
   将classes放入'./data/classes'下,命名为classes.names

2. 运行labelimg_yolov3.py将xml数据转换, 存放至'./data/label'
   train.txt: 训练文件
   test.txt: 测试文件
   data.txt: 所有文件
3. 在'./code/config.py'中更改运行参数

```
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
__M.Train.label = '.\\data\\label\\train.txt'
# __M.Train.input_size = np.random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
# input图像大小
__M.Train.input_size = np.random.choice([416])
# 训练参数: 轮次, 批次, train样本数
__M.Train.epoch = 10
__M.Train.batch_size = 6
__M.Train.sample_num = len(open(__M.Train.label,'r').readlines())
# 数据增强
__M.Train.data_aug = True
__M.Train.box_maxnum = 100 # 回收盒大小
# 模型保存地址(用于restore和save)
__M.Train.darknet_savefile = './checkpoint/darknet_yolo3/'
__M.Train.mobilenet_savefile = './checkpoint/mobilenet_yolo3/'
# 两步式训练
__M.Train.is_twostep=False
__M.Train.one_step = 10
__M.Train.two_step = 20
# variables是否可训练
__M.Train.is_training = True

#=================== Test ===================#

__M.Test =edict()
__M.Test.xml = './data/images&xml/xml'
__M.Test.label = '.\\data\\label\\test.txt'
# 是否数据增强
__M.Test.data_aug = False
# variables是否可训练
__M.Test.is_training = False

# 测试模型时使用
# 模型保存地址
__M.Test.model_savefile = './checkpoint/darknet_yolo3/'
# nms 阈值
__M.Test.nms_score = 0.45
# bbox 评分阈值
__M.Test.bboxes_score = 0.5

#=================== Export ==================#
__M.Export = edict()

# 模型版本
__M.Export.version=1
# 模型name
__M.Export.name = 'yolo3-darknet'
```

4. 运行train.py, 训练模型
5. 运行export_checkpoing.py, 将ckpt使用tf.saved_model转换,供tf.serving使用, 存放地址: './model'
6. 使用test.py测试模型
   将测试图片放入./data/test/images下,运行后, 结果会生成以图片命名的文件夹, 文件夹下放入相应的classes命名的结果图片

### 三. tornado http

1. 运行环境
   docker
   docker pull tensorflow/serving:1.13.1
   tornado

2. 运行文件 tornado_http.py
   运行参数: --port:  tornado http的运行端口(default: 8888)
                      --docker: docker 运行的IP+端口(default:127.0.0.1:8500)

3. tornado_http服务
   route1: http://ip:端口/model/coor_cls
   route2: http://ip:端口/model/array_cls
   post请求: image_path: 图片的本地地址或者网络地址
                     is_url:  True为image_path是网络地址, False为image_path是本地地址
   请求格式: json, 例如:

   ```
   image_path ='https://www.abc.com/0.jpg'             
   is_url=False
   request: {'adress':image_path,'is_url':is_url}
   ```

   返回参数:
   	    route1: 返回

           ```
   return json:
   {
       'cls_name1': # cls名
       {
           [h,w,3], 
           [h,w,3]  # 图片arrays
       },
       'cls_name2': # cls名
       {
           [h,w,3]  # 图片arrays
       }
   }
           ```

   ​       route2:返回

          ```
   return json:
   {
       'cls_name1': # cls名
       {
           [xmin,ymin,xmax,ymax],
           [xmin,ymin,xmax,ymax]  # 图片坐标
       },
       'cls_name2': # cls名
       {
           [xmin,ymin,xmax,ymax]  # 图片坐标
       }
   }
          ```

   
