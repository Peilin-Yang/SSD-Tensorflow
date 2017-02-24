from __future__ import print_function

import os, sys
import argparse
import math
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import ssd_vgg_300, ssd_common, np_methods, nets_factory
from preprocessing import ssd_vgg_preprocessing


# Main image processing routine.
def process_image(img, select_threshold=0.6, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def detect(args):
    slim = tf.contrib.slim

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NCHW'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_class = nets_factory.get_network('ssd_300_vgg')
    ssd_params = ssd_class.default_params._replace(
        num_classes=args.num_classes, 
        no_annotation_label=args.num_classes
    )
    ssd_net = ssd_class(ssd_params)
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    ckpt_filename = args.ckpt_filename
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    # Test on some demo image and visualize output.
    image_names = os.listdir(args.test_img_folder)
    #img = mpimg.imread(path + image_names[-5])
    for img in image_names:
        rclasses, rscores, rbboxes = process_image(img)
        print(rclasses, rscores, rbboxes)
        exit()

    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect images using TensorFlow SSD network')
    parser.add_argument('--checkpoint', dest='ckpt_filename', 
            help='checkpoint file path',
            required=True, type=str)
    parser.add_argument('--dir', dest='test_img_folder', 
            help='The images folder to be detected',
            required=True, type=str)
    parser.add_argument('--num_classes', dest='num_classes', 
            help='The number of classes',
            default=2, type=int)

    args = parser.parse_args()

    detect(args)

