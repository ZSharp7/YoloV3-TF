from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import random
import colorsys
from code import config
cfg = config.cfg
from code import data_pre
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def image_load(adress,model_size):
    imgs = []

    img = cv2.imread(adress)

    img = cv2.resize(img,dsize=(model_size,model_size))
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


def draw_bbox(image, bboxes,classes,is_array):
    """
    cv2截取+保存
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id]
    """
    # os.mkdir(save_path + name)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape

    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    total = {}
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        class_ind = int(bbox[5])
        try:
            if len(image[coor[1]:coor[3], coor[0]:coor[2]]) <= 0 : continue
            if classes[class_ind] not in total.keys():
                total[classes[class_ind]]=[]
            if is_array:
                total[classes[class_ind]].append(image[coor[1]:coor[3], coor[0]:coor[2]].tolist())

            else:
                total[classes[class_ind]].append(np.array([coor[0],coor[1],coor[2],coor[3]]).tolist())
            # total['images'][classes[class_ind]].append(list(image[coor[1]:coor[3], coor[0]:coor[2]]))
            # cv2.imwrite('%s%s_%d.jpg'%(save_path+name+'/',classes[class_ind],i), image[coor[1]:coor[3], coor[0]:coor[2]])
        except:
            pass

    return total

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




class Client:
    '''
    文件保存规则:
        在指定保存目录下创建以图片名命名的文件夹, 文件夹内命名方式是: 分类名_图片id.jpg
    '''
    def __init__(self):
        self.classes = data_pre.get_classes(cfg.Main.classes)
        self.output_size = 416
        self.bboxes_score = cfg.Test.bboxes_score
        self.nms_score = cfg.Test.nms_score
    def get_image(self,adress,is_array,ip):
        self.images_batch = image_load(adress, 416)

        one, two, three= self.request_server(self.images_batch, ip)
        image = cv2.imread(adress)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        pred_bbox = np.concatenate([np.reshape(one, (-1, 5 + len(self.classes))),
                                    np.reshape(two, (-1, 5 + len(self.classes))),
                                    np.reshape(three, (-1, 5 + len(self.classes)))], axis=0)
        bboxes = postprocess_boxes(pred_bbox, original_image_size, self.output_size, self.bboxes_score)
        bboxes = nms(bboxes, self.nms_score)
        total = draw_bbox(original_image, bboxes,
                          classes=self.classes,is_array=is_array)
        if len(os.listdir('./temporary/images/'))>0:
            print('删除临时文件:%s'%adress)
            os.remove(adress)
        return total
    def request_server(self,img_resized, server_url):
        '''
        用于向TensorFlow Serving服务请求推理结果的函数。
        :param img_resized: 经过预处理的图片数组，numpy array，shape：(1, 416, 416, 3)
        :param server_url: TensorFlow Serving的地址加端口，str，如：'0.0.0.0:8500'
        :return: 模型返回的结果,与export_checkpoint时一致
        '''
        # Request.
        channel = grpc.insecure_channel(server_url)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = "yolo3-darknet"  # 模型名称
        request.model_spec.signature_name = "predict"  # 签名名称
        # "images"是你导出模型时设置的输入名称

        request.inputs["images"].CopyFrom(
            tf.contrib.util.make_tensor_proto(img_resized, shape=img_resized.shape))

        response = stub.Predict(request,120.)
        channel.close()

        return response.outputs['pre_one'].float_val,response.outputs['pre_two'].float_val,response.outputs['pre_three'].float_val




