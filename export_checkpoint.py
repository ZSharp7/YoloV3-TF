import numpy as np
import tensorflow as tf
from code import data_pre
from code import config
cfg = config.cfg
from code.yolov3 import Yolo3
import os



class Export:
    def __init__(self, model_path,name,version,export_path='./model'):
        '''

        :param model_path: ckpt模型保存地址
        :param name: 转换后模型标识
        :param version: 版本
        :param export_path: 转换后存放地址
        '''
        self.classes = data_pre.get_classes(cfg.Main.classes)  # 类别
        self.anchors = data_pre.get_anchors(cfg.Main.anchors)  # anchors

        self.model_path = tf.train.get_checkpoint_state(model_path).model_checkpoint_path
        self.output_size = 416
        self.batch_size = 1
        self.export_path = export_path
        self.name = name
        self.version = version

        self.ses = tf.Session()
        self.is_training = cfg.Test.is_training
        np.random.seed(21)

        with tf.name_scope('inputs'):
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=[self.batch_size, self.output_size, self.output_size, 3],
                                               name='input_images')

        with tf.name_scope('model_create'):
            self.model = Yolo3(input_value=self.input_images, is_training=self.is_training)

        with tf.name_scope('saver_log'):
            self.saver = tf.train.Saver(tf.global_variables())

    def export(self):

        try:
            print('model is restore..')
            self.saver.restore(self.ses, self.model_path)
        except:
            print('no model_ckpt file...')
            raise ModuleNotFoundError
        export_path_base = self.export_path
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(self.version)))
        print('Exporting trained model to', export_path)

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # 输入
        tensor_info_input = tf.saved_model.utils.build_tensor_info(self.input_images)
        # 输出

        tensor_info_output1 = tf.saved_model.utils.build_tensor_info(self.model.pre_one)
        tensor_info_output2 = tf.saved_model.utils.build_tensor_info(self.model.pre_two)
        tensor_info_output3 = tf.saved_model.utils.build_tensor_info(self.model.pre_three)
        # signature
        # print(tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        # os._exit(0)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input},
                outputs={'pre_one': tensor_info_output1,'pre_two':tensor_info_output2,'pre_three':tensor_info_output3},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                # method_name=self.name))

        builder.add_meta_graph_and_variables(
            self.ses, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':prediction_signature
            })

        # export the model
        builder.save()
        print('Done exporting!')
if __name__ == '__main__':

    Export(model_path=cfg.Test.model_savefile,name=cfg.Export.name,version=cfg.Export.version).export()
