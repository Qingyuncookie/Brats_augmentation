"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
from copy import copy
from PIL import Image
pp = pprint.PrettyPrinter()
import random
from pylab import *
from skimage import transform
import cv2



get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def load_origin_label(image_path):
    img_B = norm(image_path[1])
    brain_B = np.where(img_B > 0, 4, img_B)
    real_label = np.where(brain_B > 0, image_path[2], 0)
    return real_label


def load_test_data(image_path, fine_size=256, is_testing=False):
    img_A = norm(image_path[0])
    brain_A = np.where(img_A>0, 4, img_A)
    img_B = norm(image_path[1])
    brain_B = np.where(img_B > 0, 4, img_B)
    tumor_A_cut = np.where(brain_B>0, image_path[2], 0)
    label_A = tumor_A_cut + brain_B
    label_A = label_A.astype(np.float32) / 8.
    mask_A = np.where(tumor_A_cut>0, 1, image_path[2])

    label_B = np.where(brain_B>0, image_path[2], 0)

    mask_B = np.where(image_path[3]>0, 1, image_path[3])

    real_label_A = image_path[2] + brain_A
    real_label_A = real_label_A.astype(np.float32) / 8.0

    real_label_B = image_path[3] + brain_B
    real_label_B = real_label_B.astype(np.float32) / 8.0

    # img_A = transform.resize(img_A, (fine_size, fine_size))[:, :, np.newaxis]
    # img_B = transform.resize(img_B, (fine_size, fine_size))[:, :, np.newaxis]
    # label_A = transform.resize(label_A, (fine_size, fine_size))[:, :, np.newaxis]
    # label_B = transform.resize(label_B, (fine_size, fine_size))[:, :, np.newaxis]
    # real_label_A = transform.resize(real_label_A, (fine_size, fine_size))[:, :, np.newaxis]
    # real_label_B = transform.resize(real_label_B, (fine_size, fine_size))[:, :, np.newaxis]
    # mask_A = transform.resize(mask_A, (fine_size, fine_size))[:, :, np.newaxis]
    # mask_B = transform.resize(mask_B, (fine_size, fine_size))[:, :, np.newaxis]

    img_A = cv2.resize(img_A, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    img_B = cv2.resize(img_B, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    label_A = cv2.resize(label_A, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    label_B = cv2.resize(label_B, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    real_label_A = cv2.resize(real_label_A, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    real_label_B = cv2.resize(real_label_B, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    mask_A = cv2.resize(mask_A, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]
    mask_B = cv2.resize(mask_B, (fine_size, fine_size), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]

    img_AB = np.concatenate((img_A, img_B, label_A, label_B, real_label_A, real_label_B, mask_A, mask_B), axis=2)

    return img_AB


def load_train_data(image_path, fine_size=256, is_testing=False):
    img_A = norm(image_path[0])
    brain_A = np.where(img_A>0, 4, img_A)
    img_B = norm(image_path[1])
    brain_B = np.where(img_B > 0, 4, img_B)
    tumor_A_cut = np.where(brain_B > 0, image_path[2], 0)
    label_A = tumor_A_cut + brain_B
    label_A = label_A.astype(np.float32) / 8.
    mask_A = np.where(tumor_A_cut > 0, 1, tumor_A_cut)
    label_B = image_path[3] + brain_A
    label_B = label_B.astype(np.float64) / 8.
    mask_B = np.where(image_path[3]>0, 1, image_path[3])
    # mask_D = np.where(label_D!=0, 1, label_D)
    real_label_A = image_path[2] + brain_A
    real_label_A = real_label_A.astype(np.float64) / 8.0

    real_label_B = image_path[3] + brain_B
    real_label_B = real_label_B.astype(np.float64) / 8.0

    # img_A = transform.resize(img_A, [fine_size, fine_size])[:, :, np.newaxis]
    # img_B = transform.resize(img_B, [fine_size, fine_size])[:, :, np.newaxis]
    # label_A = transform.resize(label_A, [fine_size, fine_size])[:, :, np.newaxis]
    # label_B = transform.resize(label_B, [fine_size, fine_size])[:, :, np.newaxis]
    # real_label_A = transform.resize(real_label_A, [fine_size, fine_size])[:, :, np.newaxis]
    # real_label_B = transform.resize(real_label_B, [fine_size, fine_size])[:, :, np.newaxis]
    # mask_A = transform.resize(mask_A, [fine_size, fine_size])[:, :, np.newaxis]
    # mask_B = transform.resize(mask_B, [fine_size, fine_size])[:, :, np.newaxis]

    img_A = cv2.resize(img_A, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    img_B = cv2.resize(img_B, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    label_A = cv2.resize(label_A, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    label_B = cv2.resize(label_B, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    real_label_A = cv2.resize(real_label_A, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    real_label_B = cv2.resize(real_label_B, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    mask_A = cv2.resize(mask_A, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    mask_B = cv2.resize(mask_B, (fine_size, fine_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

    img_AB = np.concatenate((img_A, img_B, label_A, label_B, real_label_A, real_label_B, mask_A, mask_B), axis=2)

    return img_AB




def norm(image):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    image = (image - image.min())/(image.max()-image.min())
    return image
# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform_(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
    # return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform_(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    # return (images+1.)/2.
    return (images - images.min()) / (images.max() - images.min())