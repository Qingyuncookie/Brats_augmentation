import argparse
import tensorflow as tf
from tensorlayer.layers import *
import tensorlayer as tl
# from __future__ import division
import os
import time
from glob import glob
from collections import namedtuple
from utils_encrip import *
import tensorflow.contrib.slim as slim
import numpy as np
from data_loader import *
from skimage import transform
import cv2

import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.set_random_seed(19)


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def discriminator(image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, s=2, name='d_h3_conv'), 'd_bn3'))
        # h3 is (16x 16 x self.df_dim*8)
        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 8, s=2, name='d_h4_conv'), 'd_bn4'))
        # h4 is (8x 8 x self.df_dim*8)
        # h5 = lrelu(instance_norm(conv2d(h4, options.df_dim * 8, s=2, name='d_h5_conv'), 'd_bn5'))
        # # h5 is (4 x 4 x self.df_dim*8)
        h5 = conv2d(h4, 1, s=1, name='d_h3_pred')
        # h4 is (4 x 4 x 1)
    return h5


def global_discriminator(image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, s=2, name='d_h3_conv'), 'd_bn3'))
        # h3 is (16x 16 x self.df_dim*8)
        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 8, s=2, name='d_h4_conv'), 'd_bn4'))
        # h4 is (8x 8 x self.df_dim*8)
        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim * 8, s=2, name='d_h5_conv'), 'd_bn5'))
        # h5 is (4 x 4 x self.df_dim*8)
        h5 = conv2d(h5, 1, s=1, name='d_h3_pred')
    # h4 is (4 x 4 x 1)
    return h5


def generator_unet(image, options, reuse=False, name="generator"):
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim * 2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim * 4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim * 8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim * 8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim * 8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim * 8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim * 8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim * 8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim * 8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim * 8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim * 8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim * 4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim * 2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        # return tf.nn.tanh(d8)
        return tf.nn.sigmoid(d8)

def generator_resnet(image, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim * 4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    # VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        tl.layers.set_name_reuse(reuse)
        print("build model started")
        rgb = tf.maximum(0.0, tf.minimum(rgb, 1.0))
        rgb = rgb * 255.0

        # rgb_scaled = rgb * 255.0
        # Convert RGB to BGR

        #        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        """
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)

            red, green, blue = tf.split(rgb_scaled, 3, 3)
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
        """
        #        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        """ input layer """
        net_in = InputLayer(rgb, name='vgg_input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        conv2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool3')
        conv3 = network
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        conv4 = network

        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        conv5_1 = network
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        conv5_2 = network
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        conv5_3 = network
        network = Conv2d(network, n_filter=512, filter_size=(3, 3),
                         strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        conv5_4 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
                            padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        #        network = FlattenLayer(network, name='flatten')
        #        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        #        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        #        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv3, conv4, conv5_3, conv5_4


def mean_squared_error(output, target, is_mean=False, name="mean_squared_error"):
    """ Return the TensorFlow expression of mean-square-error (L2) of two batch of data.

    Parameters
    ----------
    output : 2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, w, h] or [batch_size, w, h, c].
    target : 2D, 3D or 4D tensor.
    is_mean : boolean, if True, use ``tf.reduce_mean`` to compute the loss of one data, otherwise, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    """
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))

        else:
            raise Exception("Unknow dimension")
        return mse

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.frame_num = args.bbmax - args.bbmin
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.n_d = args.n_d
        self.modality_postfix = args.modality
        self.bbmin = args.bbmin
        self.bbmax = args.bbmax


        self.discriminator = discriminator
        self.global_discriminator = global_discriminator
        self.vgg_19 = Vgg19_simple_api
        self.seg_unet = generator_unet
        self.l1 = abs_criterion

        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):

        self.data_loader_A = DataLoader(data_root=self.dataset_dir, modality_postfix=self.modality_postfix,
                                        batch_size=self.batch_size, data_names = "train_A.txt")
        self.data_loader_B = DataLoader(data_root=self.dataset_dir, modality_postfix=self.modality_postfix,
                                        batch_size=self.batch_size, data_names = "train_B.txt")
        self.data_loader_valid_A = DataLoader(data_root=self.dataset_dir, modality_postfix=self.modality_postfix,
                                              batch_size=self.batch_size, data_names="train_test_A.txt")
        self.data_loader_valid_B = DataLoader(data_root=self.dataset_dir, modality_postfix=self.modality_postfix,
                                              batch_size=self.batch_size, data_names="train_test_B.txt")

        self.real_data = tf.placeholder(tf.float32,
                                        [1, self.image_size, self.image_size,
                                         self.input_c_dim * 8],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:2]
        self.brain_B = self.real_data[:, :, :, 2:3]
        self.brain_A = self.real_data[:, :, :, 3:4]
        self.real_label_A = self.real_data[:, :, :, 4:5]
        self.real_label_B = self.real_data[:, :, :, 5:6]
        self.mask_A = self.real_data[:, :, :, 6:7]
        self.mask_B = self.real_data[:, :, :, 7:8]

        self.real_A_input = tf.concat((self.real_A, self.brain_B), axis=3)
        self.fake_B = self.generator(self.real_A_input, self.options, False, name="generator")
        # self.fake_B_input = tf.concat((self.fake_B, self.real_label_A), axis=3)
        # self.fake_A_ = self.generator(self.fake_B_input, self.options, True, name="generator")

        self.mask = (1 - self.mask_B) * (1 - self.mask_A)

        self.fake_tumor_A = self.fake_B*self.mask_A
        self.real_tumor_A = self.real_A*self.mask_A
        self.fake_brain_B = self.fake_B*self.mask
        self.real_brain_B = self.real_B*self.mask


        self.vgg_real_brain_B = tf.concat((self.real_brain_B,self.real_brain_B,self.real_brain_B),axis=3)
        self.vgg_fake_brain_B = tf.concat((self.fake_brain_B,self.fake_brain_B,self.fake_brain_B),axis=3)
        self.vgg_real_tumor_A = tf.concat((self.real_tumor_A, self.real_tumor_A, self.real_tumor_A), axis=3)
        self.vgg_fake_tumor_A = tf.concat((self.fake_tumor_A, self.fake_tumor_A, self.fake_tumor_A), axis=3)
        self.net_vgg, vgg_target_B_1, vgg_target_B_2, vgg_target_B_3, vgg_target_B_4= self.vgg_19(self.vgg_real_brain_B, reuse=False)
        _, vgg_predict_B_1, vgg_predict_B_2, vgg_predict_B_3, vgg_predict_B_4 = self.vgg_19(self.vgg_fake_brain_B, reuse= True)
        self.net_vgg_, vgg_target_B_1_, vgg_target_B_2_, vgg_target_B_3_, vgg_target_B_4_ = self.vgg_19( self.vgg_real_tumor_A, reuse=True)
        _, vgg_predict_B_1_, vgg_predict_B_2_, vgg_predict_B_3_, vgg_predict_B_4_ = self.vgg_19(self.vgg_fake_tumor_A,
                                                                                            reuse=True)
        self.vgg_loss = (tl.cost.mean_squared_error(vgg_predict_B_2.outputs, vgg_target_B_2.outputs, is_mean=True) +
                         0.01 * tl.cost.mean_squared_error(vgg_predict_B_1.outputs, vgg_target_B_1.outputs,
                                                           is_mean=True) +
                         2000 * tl.cost.mean_squared_error(vgg_predict_B_1_.outputs, vgg_target_B_1_.outputs,
                                                           is_mean=True)) / 3

        self.DB_fake_input = tf.concat((self.fake_B, self.brain_B), axis=3)
        self.DB_fake = self.global_discriminator(self.DB_fake_input, self.options, reuse=False, name="discriminator")


        random_crop_origin = tf.concat((self.fake_B, self.real_B), axis=3)
        self.random_crop_after = tf.random_crop(random_crop_origin, [1, 64, 64, 2])
        self.random_crop_fake_B = self.random_crop_after[:,:,:,0:1]
        self.random_crop_real_B = self.random_crop_after[:,:,:,1:2]
        self.DB_fake_single = self.discriminator(self.random_crop_fake_B, self.options, reuse=False, name="discriminator_single")

        self.g_adv_total = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake))


        self.g_loss = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                          + self.criterionGAN(self.DB_fake_single, tf.ones_like(self.DB_fake_single))+0.001*self.vgg_loss \
                          + self.l1(self.fake_brain_B, self.real_brain_B)
        self.DB_real_input = tf.concat((self.real_B, self.real_label_B), axis=3)
        self.DB_real = self.global_discriminator(self.DB_real_input, self.options, reuse=True, name="discriminator")


        self.DB_real_single = self.discriminator(self.random_crop_real_B, self.options, reuse=True, name="discriminator_single")

        self.d_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.d_loss_fake = self.criterionGAN(self.DB_fake, tf.zeros_like(self.DB_fake))
        self.d_loss_pair = (self.d_loss_real + self.d_loss_fake) / 2
        self.d_loss_real_single = self.criterionGAN(self.DB_real_single, tf.ones_like(self.DB_real_single))
        self.d_loss_fake_single = self.criterionGAN(self.DB_fake_single, tf.zeros_like(self.DB_fake_single))
        self.d_loss_single = (self.d_loss_real_single + self.d_loss_fake_single) / 2
        self.d_loss = (self.d_loss_pair + self.d_loss_single) / 2



        self.vgg_summary = tf.summary.scalar('vgg_loss', self.vgg_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("db_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.d_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.d_loss_sum,self.d_loss_sum, self.d_loss_real_sum, self.d_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_data = tf.placeholder(tf.float32,
                                     [1, self.image_size, self.image_size,
                                      self.input_c_dim * 8], name='test')
        self.test_real_A = self.test_data[:, :, :, :self.input_c_dim]
        self.test_brain_B = self.test_data[:, :, :, 2:3]
        self.test_input = tf.concat((self.test_real_A, self.test_brain_B), axis=3)
        self.result = self.generator(self.test_input, self.options, True, name="generator")

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]

        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        self.g_optim = tf.train.AdamOptimizer(self.lr/5., beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        vgg19_npy_path = "../vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()
        params = []
        count_layers = 0
        for val in sorted(npz.items()):
            if (count_layers < 16):
                W = np.asarray(val[1][0])
                b = np.asarray(val[1][1])
                print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
                params.extend([W, b])
            count_layers += 1

        tl.files.assign_params(self.sess, params, self.net_vgg)

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        _, dataA, labelA = self.data_loader_A.load_data()
        _, dataB, labelB = self.data_loader_B.load_data()

        for epoch in range(args.epoch):
            c = list(zip(dataA, labelA))
            shuffle(c)
            dataA, labelA = zip(*c)

            c = list(zip(dataB, labelB))
            shuffle(c)
            dataB, labelB = zip(*c)


            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       labelA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       labelB[idx * self.batch_size:(idx + 1) * self.batch_size]))  # zip从ABCD中各取一个值组合成一个tuple即batch_file
                batch_images = [load_train_data(batch_file, args.fine_size) for batch_file in
                                batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                fake_B, _, g_loss, vgg_loss, summary_str = self.sess.run(
                    [self.fake_B, self.g_optim, self.g_loss, self.vgg_loss, self.g_loss_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                # Update D network

                _, d_loss, d_loss_sum = self.sess.run(
                    [self.d_optim, self.d_loss, self.d_loss_sum],
                    feed_dict={self.real_data: batch_images,
                               self.lr: lr})
                self.writer.add_summary(d_loss_sum, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f  g_loss: %4.4f d_loss: %4.4f vgg_loss：%4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, g_loss, d_loss, vgg_loss)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

            if np.mod(epoch, args.save_freq) == 0:
                self.save(args.checkpoint_dir, epoch)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):

        _, dataA, labelA = self.data_loader_valid_A.load_data()
        _, dataB, labelB = self.data_loader_valid_B.load_data()

        c = list(zip(dataA, labelA))
        shuffle(c)
        dataA, labelA = zip(*c)

        c = list(zip(dataB, labelB))
        shuffle(c)
        dataB, labelB = zip(*c)

        batch_files = list(
            zip(dataA[:self.batch_size], dataB[:self.batch_size], labelA[:self.batch_size], labelB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        # print(sample_images[0].max(), sample_images[0].min())
        sample_images = np.array(sample_images).astype(np.float32)
        fake_B = self.sess.run(
            self.fake_B,
            feed_dict={self.real_data: sample_images}
        )
        real_A = sample_images[:, :, :, 0:1]
        real_B = sample_images[:, :, :, 1:2]
        brain_B = sample_images[:, :, :, 2:3]

        merge_A = np.concatenate([real_A, brain_B, real_B, fake_B], axis=2)
        save_images(merge_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        test_A = DataLoader(data_root=self.dataset_dir, modality_postfix=self.modality_postfix,
                            batch_size=self.batch_size, data_names="train_test_B.txt")
        test_B = DataLoader(data_root=self.dataset_dir, modality_postfix=self.modality_postfix,
                            batch_size=self.batch_size, data_names="train_test_A.txt")
        patient_name_A, frame_A, label_A = test_A.load_data()
        patient_name_B, frame_B, label_B = test_B.load_data()
        print(args.checkpoint_dir)

        if self.load('./checkpoint'):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i in range(len(patient_name_A)):
            print('Processing image: ' + patient_name_A[i])
            image_path, _ = os.path.split(patient_name_A[i])
            tumor_type_path, _ = os.path.split(image_path)
            test_type_path = os.path.join(args.test_dir, os.path.basename(tumor_type_path))
            if not os.path.exists(test_type_path):
                os.mkdir(test_type_path)
            image_path = os.path.join(test_type_path, os.path.basename(image_path))+ "_aug"
            if not os.path.exists(image_path):
                os.mkdir(image_path)

            fake_imgs, labels = [], []
            for j in range(self.frame_num):
                sample_files = list(zip(frame_A[i * self.frame_num + j:i * self.frame_num + j + 1],
                                        frame_B[i * self.frame_num + j:i * self.frame_num + j + 1],
                                        label_A[i * self.frame_num + j:i * self.frame_num + j + 1],
                                        label_B[i * self.frame_num + j:i * self.frame_num + j + 1]))
                sample_image = [load_test_data(sample_file, args.fine_size) for sample_file in sample_files]
                sample_image_ = np.array(sample_image).astype(np.float32)
                fake_img = self.sess.run(self.result, feed_dict={self.test_data: sample_image_})
                fake_img = cv2.resize(fake_img.reshape(256, 256), (240, 240), interpolation=cv2.INTER_NEAREST)
                # fake_img = transform.resize(fake_img.reshape(256, 256), (240,240))
                fake_imgs.append((fake_img * 65536).astype(np.uint16))
                label = [load_origin_label(sample_file) for sample_file in sample_files]
                labels.append(label[0])
            fake_imgs = np.asarray(fake_imgs)
            labels = np.asarray(labels)
            final_fake_imgs = np.zeros([155,240,240], np.uint16)
            final_labels = np.zeros([155,240,240], np.uint8)
            self.set_ND_volume_roi_with_bounding_box_range(final_fake_imgs, self.bbmin, self.bbmax, fake_imgs)
            self.set_ND_volume_roi_with_bounding_box_range(final_labels, self.bbmin, self.bbmax, labels)
            self.save_array_as_nifty_volume(final_fake_imgs,
                                       image_path + '/' + os.path.basename(
                                           image_path) + "_" + self.modality_postfix + ".nii")
            self.save_array_as_nifty_volume(final_labels,
                                       image_path + '/' + os.path.basename(image_path) + "_seg.nii")

    def set_ND_volume_roi_with_bounding_box_range(self, volume, bb_min, bb_max, sub_volume):
        """
        set a subregion to an nd image.
        """
        out = volume
        out[np.ix_(range(bb_min, bb_max),
                   range(0, 240),
                   range(0, 240))] = sub_volume
        return out

    def save_array_as_nifty_volume(self, array, filename):
        img = sitk.GetImageFromArray(array)
        sitk.WriteImage(img, filename)




parser = argparse.ArgumentParser(description='')
# parser.add_argument('--dataset_dir', dest='dataset_dir', default='/home/li30/dataset/MICCAI_BraTS17_Data_Training',
#                     help='path of the dataset')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='/home/li/experiment/MICCAI_BraTS17_Data_Training',
                    help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--n_d', dest='n_d', type=int, default=4, help='# of discriminators')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
# parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=10,
                    help='save a model every epoch iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=1000,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True,
                    help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='../aug', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True,
                    help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--modality', dest='modality', default='t1',
                    help='the modality, such as flair, t1, t1ce, t2')
parser.add_argument('--bbmin', dest='bbmin', default=30)
parser.add_argument('--bbmax', dest='bbmax', default=110)

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)  # 使显存根据需求增长
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)


if __name__ == '__main__':
    tf.app.run()
