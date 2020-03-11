# _*_coding:utf-8_*_
import os
import xml.etree.ElementTree as ET
import shutil
import time
import numpy as np

def get_names(names_path='./data/classes/classes.names'):
    id_s = {}
    flag = 0
    with open(names_path, 'r',encoding='utf-8') as f:
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
    f = open('./data/error.txt','r')
    error = f.readlines()
    f.close()
    error = error[0].strip().split(',')

    if not os.path.exists(newdir):
        os.makedirs(newdir)
    id_s = get_names()
    flag = 0
    for fp in os.listdir(dirpath):
        # fp = 'G:\\images_Study\\yolo\data\images&xml\xml\Men-s_Loafers_&_Slip-Ons\il201903211431546734.xml'
        flag +=1
        root = ET.parse(os.path.join(dirpath, fp)).getroot()

        # root = ET.parse('G:\\images_Study\\yolo\\data\\images&xml\\xml\\Men-s_Loafers_&_Slip-Ons\\il201903211431546734.xml').getroot()
        #
        xmin, ymin, xmax, ymax = 0, 0, 0, 0

        path = root.find('path').text+' ' # 图片所在地址
        if path[-4:].strip() != 'jpg' and path[-5:].strip() != 'jpeg' and path[-4:].strip() != 'png':
            continue
        # folder = root.find("folder").text # 作为文件名保存
        if root.findall('object') == []:
            continue
        if '/'.join(path.split('\\')[5:]) == '':
            print(os.path.join(dirpath, fp))

        path='./data/images/'+ '/'.join(path.split('\\')[5:])
        # path_1 = path_1.strip()
        # dir_path = '/mnt/disk+array/download/image/jollychic/'+ '/'.join(path.split('\\')[6:])
        # dir_path = dir_path.strip()
        # if os.path.isfile(dir_path)==True:
        #     if os.path.isdir('./data/images&xml/images/'+ path.split('\\')[5]) == False:
        #         os.mkdir('./data/images&xml/images/'+ path.split('\\')[5])
        #     shutil.copyfile(dir_path,path_1)
        #     print('yes!')
        # else:
        #     print(dir_path,'not fount!')
        # path = path_1
        # path ='/mnt/disk+array/download/image/jollychic/'+ '/'.join(path.split('\\')[5:])

        for child in root.findall('object'):  # 找到图片中的所有框和cls
            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            try:
                name = child.find('name').text
                if name in error:
                    print('del: ',name)
                    continue
				
                name = id_s[child.find('name').text] # id化
            except:
                with open('./data/classes/classes.names','a+',encoding='utf-8') as names:
                    names.write(child.find('name').text+'\n')
                print('Add: ',child.find('name').text)
                id_s = get_names()
                name = id_s[child.find('name').text]  # id化

            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            if (xmax-xmin) * (ymax-ymin) <=5:
                continue
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
    label(xml_path='./data/images&xml/xml',new_path='./data/label',train_ratio=0.8)
