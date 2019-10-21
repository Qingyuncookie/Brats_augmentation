# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import os
import nibabel
import random
from scipy import ndimage
import SimpleITK as sitk
from skimage import io

class DataLoader():
    def __init__(self, data_root = None, modality_postfix=None, batch_size = 48, data_names = None ):
        """
        Initialize the calss instance
        inputs:
            config: a dictionary representing parameters
        """
        self.data_root = data_root
        self.modality_postfix = modality_postfix   # postfix:后缀
        self.label_postfix = 'seg'    # 标签后缀
        self.file_postfix  = 'nii.gz'  # 文件后缀
        self.data_names    = data_names
        self.with_ground_truth = True
        self.batch_size = batch_size


            
    def __get_patient_names(self):
        """
        get the list of patient names, if self.data_names id not None, then load patient 
        names from that file, otherwise search all the names automatically in data_root
        """
        # use pre-defined patient names
        if(self.data_names is not None):
            assert(os.path.isfile(self.data_names)) # os.path.isfill:判断路径是否为文件
            with open(self.data_names) as f:
                content = f.readlines()
            patient_names = [x.strip() for x in content]   # x.strip():去掉每行的空白
        # use all the patient names in data_root
        else:
            patient_names = os.listdir(self.data_root[0])  # os.listdir()：返回指定文件夹包含的文件或文件夹名字的列表
            patient_names = [name for name in patient_names if 'brats' in name.lower()]    # name.lower()返回所有字母为小写
        return patient_names

    def __load_one_volume(self, patient_name, mod):
        patient_dir = os.path.join(self.data_root, patient_name)    # 把目录和文件名合成一个路径
        #for bats17
        if('nii' in self.file_postfix):
            image_names = os.listdir(patient_dir)
            volume_name = None
            for image_name in image_names:
                if(mod + '.' in image_name):
                    volume_name = image_name
                    break
        # for brats15
        else:
            img_file_dirs = os.listdir(patient_dir)
            volume_name  = None
            for img_file_dir in img_file_dirs:
                if(mod+'.' in img_file_dir):
                    volume_name = img_file_dir + '/' + img_file_dir + '.' + self.file_postfix
                    break
        assert(volume_name is not None)
        volume_name = os.path.join(patient_dir, volume_name)
        volume = self.load_nifty_volume_as_array(volume_name)     #返回图片数据，以三维数组形式存储。
        return volume, volume_name

    def load_nifty_volume_as_array(self, filename, with_header=False):
        """
        load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
        The output array shape is like [Depth, Height, Width]
        inputs:
            filename: the input file name, should be *.nii or *.nii.gz
            with_header: return affine and hearder infomation
        outputs:
            data: a numpy data array
        """
        img = nibabel.load(filename)
        data = img.get_data()
        # io.imshow(data[:,:,72])
        # io.show()
        data = np.transpose(data, [2, 1, 0])  # [2,1,0]是transpose的默认转置参数，内容大小不发生变化     原本为[0,1,2],将第一个数和第三个数进行交换。
        # print(data.max())
        # io.imshow(data[72])
        # io.show()
        if (with_header):
            return data, img.affine, img.header
        else:
            return data

    def load_data(self, bbmin=30, bbmax=110):
        """
        load all the training/testing data
        """
        self.patient_names = self.__get_patient_names()
        assert(len(self.patient_names)  > 0)
        ImageNames = []
        X = []
        Y = []
        # bbox  = []
        data_num = len(self.patient_names)
        for i in range(data_num):
            volume, volume_name = self.__load_one_volume(self.patient_names[i], self.modality_postfix)

            ImageNames.append(volume_name)                             #将所有病人存放序列路径的list保存为一个list
            X.append(volume)
            label, _ = self.__load_one_volume(self.patient_names[i], self.label_postfix)
            # label = np.asarray(label > 0, np.float32)
            Y.append(label)
            #将所有病人的真值标签存到 Y 中
            if((i+1)%50 == 0 or (i+1) == data_num):
                print('Data load, {0:}% finished'.format((i+1)*100.0/data_num))


        data_all, label_all = [], []
        for i in range(len(X)):
            data_volumes = [x for x in X[i]]  # 将病人每一个序列的volume放入data_volume
            sub_data = np.asarray(data_volumes)
            label_volumes = [x for x in Y[i]]
            sub_label = np.asarray(label_volumes)

            # for frame in range(np.shape(sub_data)[0]):
            for frame in range(bbmin, bbmax):
                data_all.append(sub_data[frame, :, :])
                label_all.append(sub_label[frame, :, :])


        return ImageNames, data_all, label_all


    
