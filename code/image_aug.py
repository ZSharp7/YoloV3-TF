
from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mat
import  os
import random

def get_image_box(file_content):
    '''

    :param file_content:
    :return:
    '''
    content = file_content.strip().split(' ')
    image = np.array(cv2.imread(content[0].replace('\\', '/')))
    # image = np.array(cv2.imread(image_path))
    bbox = [list(map(lambda x: int(float(x)), i.strip().split(','))) for i in content[1:]]

    return image,bbox

def data_aug(image,bbox,size=(416,416)):
    '''
    数据增强  单张图片
    :param image:
    :param bbox:
    :param size: 目标尺寸
    :return:
    '''
    bbox_ = []
    for i in bbox:
        bbox_.append(
            ia.BoundingBox(i[0], i[1], i[2], i[3])
        )
    bbox_iaa = ia.BoundingBoxesOnImage(bbox_, shape=image.shape)
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),#随机裁剪

        iaa.Affine(translate_px=(-20,10),cval=255), # 仿真映射, 随机平移
        iaa.Resize(size, interpolation='nearest'),  # resize图像
        iaa.Fliplr(p=(1.0 if random.random() > 0.5 else 0.5)), # 水平界面翻转
        # iaa.Flipud(p=0.5) # 上下翻转
    ])
    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]
    bbox_aug = seq_det.augment_bounding_boxes([bbox_iaa])[0]
    bbox_aug_cls = []
    for id, values in enumerate(bbox_aug.to_xyxy_array()):
        value = list(values)
        for i in range(len(value)):
            if value[i]<0:
                value[i]=0
            if value[i]>size:
                value[i] = size
            value[i] = int(value[i])

        bbox_aug_cls.append(np.array([value[0],value[1],value[2],value[3],bbox[id][4]]))
    # bbox = np.array(bbox_aug_cls)
    # print(bbox_aug_cls)
    # bbox.reshape(1,len(bbox_aug_cls))
    # print(bbox.shape)
    return np.array(image_aug),bbox_aug_cls


def random_horizontal_flip( image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
    return image, bboxes
def random_crop( image, bboxes):
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
def random_translate( image, bboxes):
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

if __name__ == '__main__':
    ## 测试


    # 获取文件内容
    file_path = './data/label/train.txt'
    f = open(file_path,'r')
    file = f.readlines()
    f.close()
    # 获取classes
    file_path2 = './data/classes/classes.names'
    f = open(file_path2, 'r')
    file2 = f.readlines()
    f.close()

    for id in range(0,3):

        image,bbox = get_image_box(file[id])
        image,bbox = data_aug(image,bbox)

        # image, bbox = random_horizontal_flip(np.copy(image), np.copy(bbox))
        # # 随机裁剪
        # image, bbox = random_crop(np.copy(image), np.copy(bbox))
        # # 随机变换
        # image, bbox = random_translate(np.copy(image), np.copy(bbox))

        # 图像显示
        for i in bbox:
            print(i)
            print(file2[i[4]])
            cv2.rectangle(image, (i[0], i[1]), (i[2], i[3]), (0, 255, 255), thickness=2)
            cv2.putText(image, file2[i[4]].strip(), (i[0], i[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),
                   thickness=1)
        cv2.imshow('image%d'%id,image)
    cv2.waitKey(0)
