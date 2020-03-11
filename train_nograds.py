from code import data_pre
# from code.config import cfg

import tensorflow as tf
import numpy as np
from code.yolov3 import Yolo3
import os
import shutil
from tqdm import tqdm
import time
from code import config

cfg = config.cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
gpu_config.gpu_options.allow_growth = True


class Train:
    def __init__(self):
        # data
        self.classes = data_pre.get_classes(cfg.Main.classes)  # 类别
        self.anchors = data_pre.get_anchors(cfg.Main.anchors)  # anchors
        self.train_set = data_pre.Dataset()
        self.test_set = data_pre.Dataset('test')

        # Train
        self.model_savefile = cfg.Train.darknet_savefile if cfg.Main.backbone == 0 else cfg.Train.mobilenet_savefile
        self.epoch = cfg.Train.epoch
        self.is_twostep = cfg.Train.is_twostep
        self.one_step = cfg.Train.one_step
        self.two_step = cfg.Train.two_step
        self.batch_size = cfg.Train.batch_size
        self.is_training = cfg.Train.is_training
        self.input_size = cfg.Train.input_size
        # learning_rate
        self.first_learn_rate = 1e-4  # cos衰减学习率 ,
        self.decay_steps = 200  # 衰减步长
        self.alpha = 0.3  # alpha
        # other
        self.darknet_savefile = './checkpoint/darknet_yolo3/yolov3.ckpt'
        self.ses = tf.Session(config=gpu_config)
        np.random.seed(21)
        self.train_epoch_loss, self.test_epoch_loss = [], []
        with tf.name_scope('inputs'):
            self.input_images = tf.placeholder(dtype=tf.float32,
                                               shape=[self.batch_size, self.input_size, self.input_size, 3],
                                               name='input_images')
            self.one_bbox = tf.placeholder(dtype=tf.float32, name='one_bbox')
            self.two_bbox = tf.placeholder(dtype=tf.float32, name='two_bbox')
            self.three_bbox = tf.placeholder(dtype=tf.float32, name='three_bbox')

            self.one_recover = tf.placeholder(dtype=tf.float32, name='one_recover')
            self.two_recover = tf.placeholder(dtype=tf.float32, name='two_recover')
            self.three_recover = tf.placeholder(dtype=tf.float32, name='three_recover')
        with tf.name_scope('model_create'):
            self.model = Yolo3(input_value=self.input_images, is_training=self.is_training)
            self.load_variables = tf.global_variables()
        with tf.name_scope('loss'):
            self.giou_loss, self.conf_loss, self.pro_loss = self.model.loss_total(self.one_bbox, self.two_bbox,
                                                                                  self.three_bbox, self.one_recover,
                                                                                  self.two_recover, self.three_recover)

            self.total_loss = self.giou_loss + self.conf_loss + self.pro_loss
        with tf.variable_scope('learn_rate'):
            self.global_step = tf.Variable(0.0, dtype=tf.float32, name='global_step', trainable=False)
            self.save_step = tf.Variable(0.0, dtype=tf.float32, name='save_step', trainable=False)
            self.learn_rate = tf.train.exponential_decay(learning_rate=1e-4, global_step=self.global_step,
                                                         decay_steps=2000, decay_rate=0.9, staircase=True,
                                                         name='learning_rate')
            # self.learn_rate = 1e-4
            self.global_step = tf.assign_add(self.global_step, 1.0)
            self.save_step = tf.assign_add(self.save_step, 1.0)
        # with tf.name_scope("define_weight_decay"):
        #     moving_ave = tf.train.ExponentialMovingAverage(0.9995).apply(tf.trainable_variables())

        with tf.name_scope('optimizer'):
            # 第一部分: 主要训练分类与回归
            self.first_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')

                if var_name_mess[1] in ['conv_three', 'conv_two', 'conv_one']:
                    self.first_trainable_var_list.append(var)
            self.first_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss,
                                                                             var_list=self.first_trainable_var_list)
            # 第二部分: 整体训练
            self.second_trainable_var_list = tf.trainable_variables()
            self.second_op = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.total_loss,
                                                                                         var_list=self.second_trainable_var_list)
            # tf.keras.callbacks.ReduceLROnPlateau()

            # 梯度截断
            # optimizer = tf.train.AdamOptimizer(self.learn_rate,beta1=0.5)
            # grads, variable = zip(*optimizer.compute_gradients(self.total_loss))
            # grads, global_norm = tf.clip_by_global_norm(grads,5)
            # self.second_op = optimizer.apply_gradients(zip(grads,variable))
            # grad_op = tf.ConditionalAccumulator(dtype=tf.float32)
            # grad_op.set_global_step(self.global_step)
            # grad_op.apply_grad()

        with tf.name_scope('saver_log'):
            self.load = tf.train.Saver(self.load_variables)
            self.saver = tf.train.Saver(tf.global_variables())

            giou_loss = tf.summary.scalar("giou_loss", self.giou_loss)
            conf_loss = tf.summary.scalar("conf_loss", self.conf_loss)
            prob_loss = tf.summary.scalar("prob_loss", self.pro_loss)
            total_loss = tf.summary.scalar("total_loss", self.total_loss)
            learn_rate = tf.summary.scalar("learn_rate", self.learn_rate)

            train_epoch_loss = tf.summary.scalar('train_epoch_loss', np.mean(self.train_epoch_loss))
            test_epoch_loss = tf.summary.scalar('test_epoch_loss', np.mean(self.test_epoch_loss))

            logdir = './log/'
            # if os.path.exists(logdir): shutil.rmtree(logdir)
            # os.mkdir(logdir)

            self.batch_summary_op = tf.summary.merge([giou_loss, conf_loss, prob_loss, total_loss, learn_rate])
            self.epoch_summary_op = tf.summary.merge([train_epoch_loss, test_epoch_loss])

            self.batch_summary_writer = tf.summary.FileWriter(logdir + '/batch/', graph=self.ses.graph)
            self.epoch_summary_writer = tf.summary.FileWriter(logdir + '/epoch/', graph=self.ses.graph)

            self.batch_summary_writer.add_graph(self.ses.graph)
            self.epoch_summary_writer.add_graph(self.ses.graph)



    def train(self):
        self.ses.run(tf.global_variables_initializer())
        # self.load.restore(self.ses, self.darknet_savefile)

        try:
            ckpt = tf.train.get_checkpoint_state(self.model_savefile)
            print('model is restore in filepath:',ckpt.model_checkpoint_path)
            self.saver.restore(self.ses, ckpt.model_checkpoint_path)
            print('model is retore..')
            # self.load.restore(self.ses, self.darknet_savefile)
        except:
            print('no model_ckpt file...')
            self.ses.run(tf.global_variables_initializer())

        for epoch in range(1, 1 + self.one_step + self.two_step):

            pbar = tqdm(self.train_set)
            self.train_epoch_loss, self.test_epoch_loss = [], []

            for train_data in pbar:
                try:
                    _, giou_loss, pro_loss, conf_loss, global_step, batch_summary = self.ses.run(
                        [self.second_op, self.giou_loss, self.pro_loss, self.conf_loss, self.global_step,
                         self.batch_summary_op],
                        feed_dict={
                            self.one_bbox: train_data[0],
                            self.two_bbox: train_data[1],
                            self.three_bbox: train_data[2],
                            self.input_images: train_data[3],
                            self.one_recover: train_data[4],
                            self.two_recover: train_data[5],
                            self.three_recover: train_data[6]
                        }
                    )

                    # loss
                    train_loss = giou_loss + pro_loss + conf_loss
                    self.train_epoch_loss.append(train_loss)

                    pbar.set_description('train_loss:%.2f, giou_loss:%.2f, pro_loss:%.2f, conf_loss:%.2f' % (
                    train_loss, giou_loss, pro_loss, conf_loss))
                    self.batch_summary_writer.add_summary(batch_summary, global_step)

                    # learning_rate 自适应
                    # if global_step%1000 == 0:
                    #     length = global_step//1000
                    #     if length == 1:
                    #         length = global_step//500
                    #         if sum(self.train_epoch_loss[(length-1)*500:length*500].sort()[50]) >= sum(
                    #                 self.train_epoch_loss[(length-2)*500:(length-1)*500].sort()[
                    #                     50]):
                    #             self.learn_rate *= 0.9
                    #     else:
                    #         if sum(self.train_epoch_loss[(length-1)*1000:length*1000].sort()[100]) >= sum(
                    #                 self.train_epoch_loss[(length-2)*1000:(length-1)*1000].sort()[
                    #                     100]):
                    #             self.learn_rate *= 0.9

                except:
                    print('this train_data is error:', train_data)
                    continue

            for test_data in self.test_set:
                try:
                    test_step_loss = self.ses.run(self.total_loss, feed_dict={
                        self.one_bbox: test_data[0],
                        self.two_bbox: test_data[1],
                        self.three_bbox: test_data[2],
                        self.input_images: test_data[3],
                        self.one_recover: test_data[4],
                        self.two_recover: test_data[5],
                        self.three_recover: test_data[6]
                    })

                    self.test_epoch_loss.append(test_step_loss)

                    pbar.set_description('test_loss:%.2f' % test_step_loss)
                except:
                    print('this train_data is error:', test_data)
                    continue

            save_step_, epoch_summary = self.ses.run([self.save_step, self.epoch_summary_op])

            self.epoch_summary_writer.add_summary(epoch_summary, save_step_)

            train_epoch_loss, test_epoch_loss = np.mean(self.train_epoch_loss), np.mean(self.test_epoch_loss)
            ckpt_file = self.model_savefile + "yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.ses, ckpt_file, global_step=self.save_step)


if __name__ == '__main__':
    Train().train()
