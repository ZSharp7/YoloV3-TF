import numpy as np
import pandas as pd
import tensorflow as tf
from config import cfg
import cv2
import os
import random
import time
import threading
from image_aug import data_aug
'''
1. 将图片数据处理, 以TFRecord格式保存(多线程)
2. 使用tf.data.Dateset进行读取
'''

def get_classes(file_path):
    '''
    从文件中, 得到cls
    :param file_path:
    :return: list
    '''
    result = {}
    with open(file_path, 'r') as f:
        # print(f.readlines())
        for id, value in enumerate(f.readlines()):
            result[id] = value.replace('\n', '')
    return result

def get_anchors(file_path):
    '''
    从文件中得到anchors
    :param file_path:
    :return: list
    '''
    with open(file_path, 'r') as f:
        content = f.readline()
    content = content.strip().split(sep=',')
    content = np.array([float(i) for i in content]).reshape([-1, 2])
    return content
class Dataset:
    def __init__(self, type='train'):
        self.type = type
        self.classes = get_classes(cfg.Main.classes) # 类别
        self.anchors = get_anchors(cfg.Main.anchors) # anchors
        self.label_path = cfg.Train.label if type == 'train' else cfg.Test.label

        self.batch_size1 = cfg.Train.batch_size # 批次尺寸
        self.batch_size = 1
        self.strides = np.array(cfg.Main.strides) # 缩放大小
        self.data_aug = cfg.Train.data_aug if type =='train' else cfg.Test.data_aug
        self.epoch = 0
        self.sample_num = len(os.listdir('./data/label/'+type+'/'))
        self.box_maxnum = 150
        self.startid = 0
        self.endid = 0
        f = open(self.label_path, 'r')
        self.label_file = f.readlines()
        np.random.shuffle(self.label_file)
        f.close()
        self.flag = True
    def image_preporcess(self, image, target_size, gt_boxes=None):
        '''
        1. 读取图片
        2. 将图片缩放, 框坐标随之改变
        3. 预留图像增强
        :param image:
        :param target_size:
        :param gt_boxes:
        :return: 图像和框
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image = cv2.imread(image)
        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes

    def get_image_box(self, file_content):
        '''
        从文件中读取图像和框位置
        :param file_content:
        :return:
        '''
        image = ''
        bbox = []
        content = file_content.strip().split(' ')
        image = np.array(cv2.imread(content[0].replace('\\', '/')))
        # image = np.array(cv2.imread(image_path))
        bbox = [list(map(lambda x: int(float(x)), i.strip().split(','))) for i in content[1:]]
        # print(bbox)
        # os._exit(0)
        # 图像增强
        # if self.data_aug:  # 是否图像增强
        #     # 随机水平翻转
        #     image, bbox = self.random_horizontal_flip(np.copy(image), np.copy(bbox))
        #     # 随机裁剪
        #     image, bbox = self.random_crop(np.copy(image), np.copy(bbox))
        #     # 随机变换
        #     image, bbox = self.random_translate(np.copy(image), np.copy(bbox))
        #
        # # 缩放图片,定位盒子
        # image, bbox = self.image_preporcess(image=image, target_size=[self.train_input_size, self.train_input_size], gt_boxes=np.array(bbox))

        # 图像增强+ resize
        image,bbox = data_aug(image,bbox,size=self.train_input_size)

        return image, bbox
    # 图像增强
    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
    def iou(self, box1, box2):
        '''
        [x,y,w,h]
        iou
        :param box1:
        :param box2:
        :return:
        '''
        # [x1,y1,x2,y2]
        box1 = np.array(box1, np.float)
        box2 = np.array(box2, np.float)
        box1_left_up = box1[:2] - box1[2:] * 0.5
        box1_right_down = box1[:2] + box1[2:] * 0.5
        box2_left_up = box2[:2] - box2[2:] * 0.5
        box2_right_down = box2[:2] + box2[2:] * 0.5

        x1 = max(box1_left_up[0], box2_left_up[0])
        y1 = max(box1_left_up[1], box2_left_up[1])
        x2 = min(box1_right_down[0], box2_right_down[0])
        y2 = min(box1_right_down[1], box2_right_down[1])
        area = (x1 - x2) * (y1 - y2)

        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        return area / (box1_area + box2_area - area)


    def anchor_gt(self, bboxes):
        '''
        anchor与gt匹配
        :param bboxes: 一个图片的所有bbox
        :return: 返回匹配的3个结果和 后续区分负样本是否有可能变成正样本(bbox的xywh和数量)
        '''

        total = [np.zeros((self.train_output_size[i], self.train_output_size[i], 3,
                           5 + len(self.classes))) for i in range(3)]
        bboxes_xywh = [np.zeros((self.box_maxnum,4)) for i in range(3)]
        bbox_count = np.zeros((3,))
        # 初始化:  输入尺寸的h,w + 3个bool类型的尺度判断 + 5(x,y,w,h,1.0) + cls
        for bbox in bboxes:
            bbox_xy = bbox[:4]  # bbox坐标 [xmin,ymin,xmax,ymax]

            bbox_id = np.zeros((len(self.classes),), np.float)
            bbox_id[bbox[4]] = 1  # bbox   cls
            uniform_distribution = np.full(len(self.classes), 1.0 / len(self.classes))
            deta = 0.01
            bbox_id = bbox_id * (1 - deta) + deta * uniform_distribution  # smooth

            bbox_xywh = np.concatenate([(bbox_xy[:2] + bbox_xy[2:]) * 0.5, bbox_xy[2:] - bbox_xy[:2]],
                                       axis=-1)  # 将坐标转换成xywh
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]  # 按照strides进行三个尺度的缩放


            has_true = False
            iou_list = []
            for id in range(0,3):
                anchors_xywh = np.zeros((3, 4))

                anchors_xywh[:, :2], anchors_xywh[:, 2:] = np.floor(bbox_xywh_scaled[id, :2]) + 0.5, self.anchors[id*3:id*3+3]

                bbox_iou = np.array([self.iou(anchors_xywh[i],bbox_xywh_scaled[i]) for i in range(len(anchors_xywh))])  # 计算iou值,进行分配
                iou_list.append(bbox_iou)
                bbox_iou_is = bbox_iou>0.3

                if  np.any(bbox_iou_is):
                    x,y = np.floor(bbox_xywh_scaled[id,:2]).astype(np.int32)

                    total[id][y, x, bbox_iou_is, :] = 0
                    total[id][y, x, bbox_iou_is, 0:4] = bbox_xywh
                    total[id][y, x, bbox_iou_is, 4:5] = 1.0
                    total[id][y, x, bbox_iou_is, 5:] = bbox_id

                    # 回收
                    bboxes_xywh_id = int(bbox_count[id]%self.box_maxnum)
                    bboxes_xywh[id][bboxes_xywh_id,:4] = bbox_xywh
                    bbox_count[id] += 1



                    has_true = True # 判断是否全部超出阈值范围

            if not has_true:
                # 自动划入阈值最大的集合中
                max_iou_id = int(np.argmax(np.array(iou_list).reshape(-1),axis=-1))
                best_detect = int(max_iou_id / 3)
                best_anchor = int(max_iou_id % 3)
                x, y = np.floor(bbox_xywh_scaled[best_detect, :2]).astype(np.int32)
                total[best_detect][y, x, best_anchor, :] = 0
                total[best_detect][y, x, best_anchor, 0:4] = bbox_xywh
                total[best_detect][y, x, best_anchor, 4:5] = 1.0
                total[best_detect][y, x, best_anchor, 5:] = bbox_id

                # 回收
                bboxes_xywh_id = int(bbox_count[best_detect] % self.box_maxnum)
                bboxes_xywh[best_detect][bboxes_xywh_id, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        one_bbox,two_bbox,three_bbox = total
        one_recover,two_recover,three_recover = bboxes_xywh

        return one_bbox,two_bbox,three_bbox,one_recover,two_recover,three_recover





    def __len__(self):
        return int(np.ceil(self.sample_num/self.batch_size))

    def data_one(self,data):
        '''
        将单张图片的图片数据和bbox数据进行处理, 并返回
        :param data:
        :return:
        '''
        # self.train_input_size = np.random.choice(cfg.Train.input_size)
        # self.train_input_size = 416
        self.train_input_size = cfg.Train.input_size
        self.train_output_size = self.train_input_size // self.strides

        # 初始化容器
        ## 匹配的框
        batch_one_bbox = np.zeros((self.batch_size,self.train_output_size[0],self.train_output_size[0],3,5+len(self.classes)))
        batch_two_bbox = np.zeros([self.batch_size,self.train_output_size[1],self.train_output_size[1],3,5+len(self.classes)])
        batch_three_bbox = np.zeros([self.batch_size,self.train_output_size[2],self.train_output_size[2],3,5+len(self.classes)])
        ## 图片
        batch_image = np.zeros([self.batch_size,self.train_input_size,self.train_input_size,3])
        ## 回收器
        batch_one_recover = np.zeros([self.batch_size,self.box_maxnum,4])
        batch_two_recover = np.zeros([self.batch_size,self.box_maxnum,4])
        batch_three_recover = np.zeros([self.batch_size,self.box_maxnum,4])
        label_file = self.label_file


        image, bboxes = self.get_image_box(file_content=data)
        one_bbox,two_bbox,three_bbox,one_recover,two_recover,three_recover = self.anchor_gt(bboxes)
        batch_one_bbox[:,:,:,:] = one_bbox
        batch_two_bbox[:,:,:,:] = two_bbox
        batch_three_bbox[:,:,:,:] = three_bbox
        batch_image[:,:,:] = image
        batch_one_recover[:,:] = one_recover
        batch_two_recover[:,:] = two_recover
        batch_three_recover[:,:] = three_recover

        return batch_one_bbox, batch_two_bbox, batch_three_bbox, batch_image, batch_one_recover, batch_two_recover, batch_three_recover


    def data_total(self,file,name):
        '''
        将单张图片处理后的数据, 以TFRecord格式保存
        :param file: 单张图片的数据, 图片地址+bbox(xmin,ymin,xmax,ymax,cls)
        :param name:
        :return:
        '''
        type = self.type
        value = self.data_one(file)
        record_writer = tf.python_io.TFRecordWriter('./data/label/'+type+'/data_' + str(name) + '.tfrecords')
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'one_box': tf.train.Feature(float_list=tf.train.FloatList(value=value[0].flatten())),
                'two_box': tf.train.Feature(float_list=tf.train.FloatList(value=value[1].flatten())),
                'three_box': tf.train.Feature(float_list=tf.train.FloatList(value=value[2].flatten())),
                'image': tf.train.Feature(float_list=tf.train.FloatList(value=value[3].flatten())),
                'one_recover': tf.train.Feature(float_list=tf.train.FloatList(value=value[4].flatten())),
                'two_recover': tf.train.Feature(float_list=tf.train.FloatList(value=value[5].flatten())),
                'three_recover': tf.train.Feature(float_list=tf.train.FloatList(value=value[6].flatten()))
            }
        ))

        record_writer.write(example.SerializeToString())
        if name % 100 == 0 and name != 0:
            print('%d is done !' % (name))





    def load_imagebbox(self,file):
        '''
        从TFRecord文件中, 取出数据, 用于Dataset的map方法
        :param file:
        :return: Tensor (type: tuple)
        '''
        features = tf.parse_single_example(
            file,
            features={
                'one_box': tf.VarLenFeature(dtype=tf.float32),
                'two_box': tf.VarLenFeature(dtype=tf.float32),
                'three_box': tf.VarLenFeature(dtype=tf.float32),
                'image': tf.VarLenFeature(dtype=tf.float32),
                'one_recover': tf.VarLenFeature(dtype=tf.float32),
                'two_recover': tf.VarLenFeature(dtype=tf.float32),
                'three_recover': tf.VarLenFeature(dtype=tf.float32)
            }
        )
        one_box = features['one_box'].values
        two_box = features['two_box'].values
        three_box = features['three_box'].values
        image = features['image'].values
        one_recover = features['one_recover'].values
        two_recover = features['two_recover'].values
        three_recover = features['three_recover'].values
        return one_box,two_box,three_box,image,one_recover,two_recover,three_recover

    def thread(self, num):
        '''
        多线程处理数据
        :param num:  线程数
        :return:
        '''
        self.nowtime = time.time()
        type = self.type
        print('The ' + type + ' data is saving ...')
        file_path = './data/label/' + type + '.txt'
        f = open(file_path, 'r')
        file = f.readlines()
        f.close()
        length = len(file)
        num_2 = num
        times = length // num if length % num == 0 else (length // num) + 1
        for i in range(times):
            threads = []
            if i == length // num and times > length // num:
                num_2 = length % num
            for j in range(num_2):
                threads.append(threading.Thread(target=self.data_total(file[(i * num) + j], (i * num) + j), name='one_%d' % j))
                threads[j].start()
            for j in threads:
                j.join()
        print('用时: ',time.time() - self.nowtime)
    def bulid_data(self):
        '''
        搭建tf.data.Dataset
        :return:
        '''
        tfrecordfile = tf.gfile.Glob('./data/label/'+self.type+'/*.tfrecords')
        dataset = tf.data.TFRecordDataset(filenames=tfrecordfile)
        dataset = dataset.map(self.load_imagebbox)
        dataset = dataset.repeat().batch(self.batch_size1).prefetch(self.batch_size1)
        return dataset
if __name__ == '__main__':

    # data = Dataset('train')
    #
    # data.thread(1000)
    pass
    '''
    使用多线程+SSD
    SSD花费时间: 915.7977125644684
                10000(进程数) 899.5327935218811
    100(101条数据): 35.38681483268738
    30(101条数据): 36.35931062698364
    2(101条数据): 44.1765832901001
    没用多线程
    # HDD花费时间: 1337.8589267730713
    # SSD花费时间: 973.7142026424408
    '''
    '''
        TFRecord数据存储维度
        (1, 52, 52, 3, 65)
        (1, 26, 26, 3, 65)
        (1, 13, 13, 3, 65)
        (1, 416, 416, 3)
        (1, 150, 4)
        (1, 150, 4)
        (1, 150, 4)
    '''
