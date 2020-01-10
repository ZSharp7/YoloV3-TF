# _*_coding:utf-8_*_
import os
import xml.etree.ElementTree as ET
from code.config import cfg
import time
import numpy as np

def get_names(names_path='./data/classes/Men-s_T-shirts__cls.names'):
    id_s = {}
    flag = 0
    with open(names_path, 'r') as f:
        for id in f.readlines():
            id_s[id[:-1]] = flag
            flag += 1
    return id_s
def get_newlabel(dirpath='./data/images&xml/xml/Men-s_T-shirts',newdir='./data/label/Men-s_T-shirts'):
    '''
    将labelimg生成的xml转换为yolo3需要的label
    :param dirpath: 存放xml的目录
    :param newdir: 输出目录
    :return:
    '''
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    id_s = get_names()
    flag = 0
    for fp in os.listdir(dirpath):
        # fp = 'G:\\images_Study\\YOLOv3-tensorflow\data\images&xml\xml\Men-s_Loafers_&_Slip-Ons\il201903211431546734.xml'
        flag +=1
        root = ET.parse(os.path.join(dirpath, fp)).getroot()

        # root = ET.parse('G:\\images_Study\\YOLOv3-tensorflow\\data\\images&xml\\xml\\Men-s_Loafers_&_Slip-Ons\\il201903211431546734.xml').getroot()
        #
        xmin, ymin, xmax, ymax = 0, 0, 0, 0

        path = root.find('path').text+' ' # 图片所在地址
        if path[-4:].strip() != 'jpg' and path[-5:].strip() != 'jpeg':
            continue
        # folder = root.find("folder").text # 作为文件名保存
        if root.findall('object') == []:
            continue
        for child in root.findall('object'):  # 找到图片中的所有框和cls

            sub = child.find('bndbox')  # 找到框的标注值并进行读取


            try:
                name = id_s[child.find('name').text] # id化
            except:
                with open('./data/classes/Men-s_T-shirts__cls.names','a+') as names:
                    names.write(child.find('name').text+'\n')
                print('Add: ',child.find('name').text)
                id_s = get_names()
                name = id_s[child.find('name').text]  # id化

            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            # try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            #     x_center = (xmin + xmax) / (2 * width)
            #     y_center = (ymin + ymax) / (2 * height)
            #     w = (xmax - xmin) / width
            #     h = (ymax - ymin) / height
            # except ZeroDivisionError:
            #     print(filename, '的数值有问题')
            path += ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(name)+' ']) # 拼接成一行,并保存

        with open(os.path.join(newdir, 'data.txt'), 'a+') as f:
            f.write(path[:-1] + '\n')

def label(xml_path,new_path,train_ratio=0.8):
    for i in os.listdir(xml_path):
        get_newlabel(dirpath='./data/images&xml/xml/'+i,newdir=new_path)
    f = open(new_path+'/data.txt','r')
    f_data = f.readlines()
    np.random.shuffle(f_data)
    f.close()
    f_train = open(new_path+'/train.txt','w')
    f_test = open(new_path+'/test.txt','w')

    f_train.write(''.join(f_data[:int(len(f_data)*train_ratio)]).strip())
    f_test.write(''.join(f_data[int(len(f_data)*train_ratio):]).strip())

    f_train.close()
    f_test.close()



if __name__ == '__main__':
    label(xml_path='./data/images&xml/xml',new_path='./data/label')

