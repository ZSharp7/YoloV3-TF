from code import darknet
import tensorflow as tf
import os
import time
import random
import cv2
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_config = tf.ConfigProto(allow_soft_placement=True)
# gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_config.gpu_options.allow_growth = True


# class VariablesUpdate:
#     def __init__(self):
#         self.train_total_loss = tf.Variable(0.0, tf.float32)
#         self.test_total_loss = tf.Variable(0.0, tf.float32)
#         self.test_total_acc = tf.Variable(0.0, tf.float32)
#     def update(self,loss,acc):
#         train_loss_update = tf.assign_add(self.train_total_loss, loss)
#         test_loss_update = tf.assign_add(self.test_total_loss, loss)
#         test_acc_update = tf.assign_add(self.test_total_acc, acc)
#     def get_variables(self):
#         return
class Darknet:
    def __init__(self, batch_size, epoch, grads_size, image_size, is_training=True):
        self.is_training = is_training
        self.batch_size = batch_size
        self.epoch_size = epoch
        # 累积梯度step
        self.grads_size = grads_size
        # darknet.weights 提取后文件夹位置
        self.weights_file = ''
        self.train_data = pd.read_csv('./data/label/darknet_train.csv')
        self.test_data = pd.read_csv('./data/label/darknet_test.csv')
        # 图片尺寸
        self.image_size = image_size

    def model(self, input_data, is_training):
        _, _, input_data = darknet.backbone(input_data, is_training)  # 1,13,13,1024
        # 全局平均池化
        input_data = tf.keras.layers.GlobalAveragePooling2D().apply(input_data)
        # input_data = tf.layers.average_pooling2d(inputs=input_data,pool_size=13,strides=13,padding='same')
        input_data = tf.layers.dense(inputs=input_data, units=399, activation=None,
                                     bias_initializer=tf.zeros_initializer())
        input_data = tf.reshape(input_data, (-1, 399))
        return input_data

    def gradients_add(self, optimizer, loss, var_list):
        '''
         梯度累积
        :param optimizer: 优化器
        :param loss: 损失
        :param var_list: 参数
        :return:
        '''
        with tf.variable_scope('gradient'):
            gradient_all = optimizer.compute_gradients(loss, var_list=var_list)  # gradient of network (with NoneType)
            grads_vars = [v for (g, v) in gradient_all if g is not None]  # all variable that has gradients
            gradient = optimizer.compute_gradients(loss, grads_vars)  # gradient of network (without NoneType)
            grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
                            for (g, v) in gradient]

        train_op = optimizer.apply_gradients(grads_holder)

        return train_op, gradient, grads_holder

    def run(self):
        self.data_aug = True
        batch_size = self.batch_size
        train_or_test = tf.placeholder(tf.int32, shape=[], name='train_or_test')
        dataset = tf.data.Dataset.from_generator(self.get_data, args=(train_or_test,),
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([self.image_size, self.image_size, 3],
                                                                [399]))
        with tf.variable_scope('input_data'):
            dataset = dataset.batch(batch_size).prefetch(32)
            iterator = dataset.make_initializable_iterator()
            one_batch = iterator.get_next()
            x_input, y_input = one_batch
        # model
        out_data = self.model(x_input, self.is_training)
        with tf.variable_scope('loss'):

            global_step = tf.Variable(0.0, tf.float32)
            global_step_update = tf.assign_add(global_step, 1.0)

            y_label = tf.argmax(y_input, axis=1)
            y_hat = tf.argmax(out_data, axis=1)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_input, logits=out_data))
            acc = tf.reduce_mean(tf.cast(tf.equal(y_label, y_hat), dtype=tf.float32))
            learn_rate = tf.train.exponential_decay(learning_rate=0.003, global_step=global_step,
                                                    decay_steps=3000, decay_rate=0.9, staircase=True,
                                                    name='learning_rate')

            train_total_loss = tf.Variable(0.0, tf.float32)
            test_total_loss = tf.Variable(0.0, tf.float32)
            test_total_acc = tf.Variable(0.0, tf.float32)

            train_loss_update = tf.assign_add(train_total_loss, loss)
            test_loss_update = tf.assign_add(test_total_loss, loss)
            test_acc_update = tf.assign_add(test_total_acc, acc)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            # train_op = optimizer.minimize(loss=loss,var_list=tf.trainable_variables())
            # 梯度累积
            train_op, gradient, grads_holder = self.gradients_add(optimizer=optimizer, loss=loss,
                                                                  var_list=tf.trainable_variables())
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([train_op, global_step_update]):
                    op = tf.no_op()
        with tf.variable_scope('save_summary'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('acc', acc)
            tf.summary.scalar('learn_rate', learn_rate)
            summary_op = tf.summary.merge_all()
            self.load_variables = [i for i in tf.trainable_variables() if 'darknet' in i.name]

            load = tf.train.Saver(self.load_variables, name='pre_train_saver')
            saver = tf.train.Saver(tf.global_variables(), name='train_saver')

        with tf.Session(config=gpu_config) as ses:
            summary_write = tf.summary.FileWriter('./log/darknet/', graph=ses.graph)
            tf.global_variables_initializer().run()
            # 是否有模型文件
            ckpt = tf.train.get_checkpoint_state('./checkpoint/darknet/')
            if ckpt:
                saver.restore(ses, ckpt.model_checkpoint_path)
                print('train_model is load.')
            else:
                # 载入预训练权重
                load.restore(ses, './checkpoint/darknet2/darknet.ckpt')
                print('pre_train_model is load.')

            for i in range(self.epoch_size):
                self.train(ses, iterator, train_or_test, batch_size, loss, acc, summary_op, global_step_update,
                           gradient, train_loss_update,
                           summary_write, global_step, grads_holder,op)
                self.test(ses, iterator, train_or_test, batch_size, test_loss_update, test_acc_update)

                train_loss, test_loss, test_acc = ses.run([train_total_loss, test_total_loss, test_total_acc])
                print('test_total_loss: %.4f, test_total_acc: %.4f' % (
                    test_loss / len(self.test_data), test_acc / len(self.test_data)))
                save_path = './checkpoint/darknet/darknet_train%.4f_test%.4f.ckpt' % \
                            (train_loss / len(self.train_data), test_loss / len(self.test_data))
                saver.save(sess=ses, save_path=save_path, global_step=global_step)

    def test(self, ses, iterator, train_or_test, batch_size, test_loss_update, test_acc_update):
        # 初始化dataset迭代器
        ses.run(iterator.initializer, feed_dict={train_or_test: 0})
        # 显示信息
        length = len(self.test_data)
        all_step = [length // batch_size + 1 if length % batch_size != 0 else length // batch_size][0]
        print('[%d]test is run.' % all_step)
        while True:
            try:
                _, _ = ses.run([test_loss_update, test_acc_update])
            except:
                break

    def train(self, ses, iterator, train_or_test, batch_size, loss, acc, summary_op, global_step_update, gradient,
              train_loss_update,
              summary_write, global_step, grads_holder,op):
        # 初始化dataset迭代器
        ses.run(iterator.initializer, feed_dict={train_or_test: 1})
        # 显示信息
        length = len(self.train_data)
        all_step = [length // batch_size + 1 if length % batch_size != 0 else length // batch_size][0]
        step = 1

        # 梯度存储
        grads = []
        grads_flag = 1
        while True:
            try:
                loss_, acc_, summary_, _, grad, _ = \
                    ses.run([loss, acc, summary_op, global_step_update, gradient, train_loss_update])
                if step % (self.grads_size + 1) == 0:
                    print('[%d/%d]train_loss: %.4f, acc: %.4f' % (step, all_step, loss_, acc_))
                summary_write.add_summary(summary_, ses.run(global_step))
                step += 1
                grads.append(grad)
                # 梯度更新
                if grads_flag % self.grads_size == 0:
                    grads_sum = {}
                    for i in range(len(grads_holder)):
                        k = grads_holder[i][0]
                        # tf.clip_by_global_norm(grads,5)
                        grads_sum[k] = sum([g[i][0] for g in grads])
                    _ = ses.run(op, feed_dict=grads_sum)
                    grads = []
                grads_flag += 1
                # tf.get_default_graph().finalize()
            except:
                break

    def get_data(self, train_or_test):
        if train_or_test == 1:
            data = self.train_data
            data_aug = True
            self.is_training = True
        else:
            data = self.test_data
            data_aug = False
            self.is_training = False
        data = data.sample(frac=1.0).reset_index(drop=True)
        for i in range(len(data)):
            content = data.iloc[i]
            y = [0.0 for i in range(399)]
            y[int(content['cls'])] = 1.0
            box = [int(float(i)) for i in content['box'].strip().split(',')]
            try:
                img = cv2.imread(content['img_path'])
                x = cv2.resize(img[box[1]:box[3], box[0]:box[2]], (self.image_size, self.image_size))
            except:
                print('error: ', content)
                continue

            if data_aug:
                # 随机亮度 + 对比度
                x = self.random_bright(x)
                # 随机水平翻转
                x = self.random_horizontal_flip(x)
                # 随机裁剪
                x = self.random_crop(x)
                # 随机变换
                x = self.random_translate(x)
            x = x / 255.0

            yield x.tolist(), y

    def random_bright(self, image):
        # 1 随机亮度
        '''
        if random.random() > 0.5:
            min=0.5, max=2.0
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            random_br = np.random.uniform(min, max)
            mask = hsv[:, :, 2] * random_br > 255
            v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
            hsv[:, :, 2] = v_channel
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        '''

        # 2 随机对比度, 随机亮度
        # alpha * image + beta
        if random.random() > 0.5:
            new_image = np.zeros(image.shape, image.dtype)
            alpha = np.random.uniform(1.3, 1.6)
            beta = np.random.uniform(0.5, 2.0)

            for h in range(image.shape[0]):
                for w in range(image.shape[1]):
                    for c in range(image.shape[2]):
                        new_image[h, w, c] = np.clip(alpha * image[h, w, c] + beta, 0, 255)
            return new_image
        else:
            return image

    def random_horizontal_flip(self, image):

        if random.random() < 0.5:
            image = image[:, ::-1, :]
        return image

    def random_crop(self, image):

        if random.random() < 0.5:
            h, w, _ = image.shape
            new_image = np.zeros(image.shape, image.dtype)
            y_max = random.randint(0, 10)
            x_max = random.randint(0, 10)
            image = image[0:h - y_max, 0:w - x_max]

            for h in range(image.shape[0]):
                for w in range(image.shape[1]):
                    for c in range(image.shape[2]):
                        new_image[h, w, c] = image[h, w, c]
            return new_image
        return image

    def random_translate(self, x):

        if random.random() < 0.5:
            r, c = x.shape[:2]
            m = cv2.getRotationMatrix2D((c / np.random.randint(1, 4), r / np.random.randint(1, 4)),
                                        np.random.randint(-10, 11), 1)
            x = cv2.warpAffine(x, m, (c, r))
        return x


def data_pre(path='./data/label/train.txt'):
    f = open(path, 'r')
    data = f.readlines()
    f.close()
    data_label = []
    flag = 0
    for i in data:
        # print(flag)
        flag += 1
        content = i.strip().split(' ')
        img_path = content[0]
        bbox = content[1:]
        for j in bbox:
            box = j.strip().split(',')
            data_label.append({
                'img_path': img_path,
                'box': ','.join(box[:-1]),
                'cls': box[-1]
            })
    pd.DataFrame(data_label).to_csv('./data/label/darknet_train.csv', index=None)


if __name__ == '__main__':
    # 数据处理
    # data_pre()
    # 开始训练
    Darknet(batch_size=48, epoch=20, grads_size=5, image_size=224).run()
    '''
    batch_size : 批次尺寸
    epoch: 轮次尺寸
    grads_size: 累积梯度step
    image_size: 网络输入图片的尺寸
    '''
