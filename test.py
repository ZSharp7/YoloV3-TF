import numpy as np
import tensorflow as tf
from PIL import Image
from code import data_pre
from code.config import cfg
from code.yolov3 import Yolo3
import random
import cv2
import colorsys
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont

def read_class_names(class_file_name):
    '''loads class name from a file'''

    # ID:value
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def image_load(img_names,model_size):
    imgs = []

    for img_name in img_names:
        img = Image.open(img_name)
        img = img.resize(size=(model_size,model_size))
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img[:, :, :3], axis=0)
        imgs.append(img)

    imgs = np.concatenate(imgs)

    return imgs

def bboxes_iou(boxes1, boxes2):
    # iou  boxes
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def draw_bbox(image, bboxes, classes, show_label=True):
    """
    cv2画出来, 并show
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id]
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    #  (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 取出超出范围的盒子
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 去除无效盒子
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 去除分数低的盒子
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)






class Test:
    def __init__(self,images_path,model_path,output_size):
        self.classes = data_pre.get_classes(cfg.Main.classes)  # 类别
        self.anchors = data_pre.get_anchors(cfg.Main.anchors)  # anchors
        self.images_path = images_path
        self.batch_size = len(self.images_path)
        self.model_path = tf.train.get_checkpoint_state(model_path).model_checkpoint_path
        # self.model_path = './checkpoint/darknet/yolov3.ckpt'
        self.output_size = output_size
        self.ses = tf.Session()
        self.is_training = cfg.Test.is_training
        np.random.seed(21)

        with tf.name_scope('inputs'):
            self.input_images = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.output_size,self.output_size,3], name='input_images')

        with tf.name_scope('model_create'):
            self.model = Yolo3(input_value=self.input_images,is_training=self.is_training)

        with tf.name_scope('saver_log'):
            self.saver = tf.train.Saver(tf.global_variables())
    def test(self):
        images_batch = image_load(self.images_path, self.output_size)
        original_image = cv2.imread(self.images_path[2])
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        try:
            print('model is restore..')
            self.saver.restore(self.ses, self.model_path)
        except:
            print('no model_ckpt file...')
            self.ses.run(tf.global_variables_initializer())
        # bboxes = []
        bboxes = self.ses.run([self.model.pre_one,self.model.pre_two,self.model.pre_three], feed_dict={self.input_images: images_batch})
        pred_bbox = np.concatenate([np.reshape(bboxes[2][0], (-1, 5 + len(self.classes))),
                                    np.reshape(bboxes[2][1], (-1, 5 + len(self.classes))),
                                    np.reshape(bboxes[2][2], (-1, 5 + len(self.classes)))], axis=0)
        bboxes = postprocess_boxes(pred_bbox, original_image_size, self.output_size, 0.2)
        bboxes = nms(bboxes, 0.45)
        image = draw_bbox(original_image, bboxes, classes=self.classes)




        image = Image.fromarray(image)
        image.show()

        # draw_boxes(self.images_path, pred_bbox, self.classes, (self.output_size,self.output_size))
if __name__ == '__main__':
    images = ['./data/test/images/1.jpg','./data/test/images/2.jpg','./data/test/images/3.jpg',]
    Test(images,output_size=416,model_path=cfg.Test.model_savefile).test()


