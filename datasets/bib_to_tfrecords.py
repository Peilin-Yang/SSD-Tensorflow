# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
from __future__ import print_function
import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.bib_common import BIB_LABELS

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
IMAGE_SHAPE =[500, 500, 3]

def _process_image(directory, name, dataset='training'):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    if dataset == 'training':
        filename = os.path.join(directory, DIRECTORY_IMAGES, dataset, name.split('-')[0], name + '.jpg')
    else:
        filename = os.path.join(directory, DIRECTORY_IMAGES, dataset, name + '.jpg')

    image_data = tf.gfile.FastGFile(filename, 'r').read()
    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, dataset, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(BIB_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        ymin = max(min(float(bbox.find('ymin').text), IMAGE_SHAPE[0]), 0)
        xmin = max(min(float(bbox.find('xmin').text), IMAGE_SHAPE[1]), 0)
        ymax = max(min(float(bbox.find('ymax').text), IMAGE_SHAPE[0]), 0)
        xmax = max(min(float(bbox.find('xmax').text), IMAGE_SHAPE[1]), 0)
        if ymax > 500 or xmax > 500:
            print(name)
        bboxes.append((ymin / IMAGE_SHAPE[0],
                       xmin / IMAGE_SHAPE[1],
                       ymax / IMAGE_SHAPE[0],
                       xmax / IMAGE_SHAPE[1]
                       ))
    return image_data, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(IMAGE_SHAPE[0]),
            'image/width': int64_feature(IMAGE_SHAPE[1]),
            'image/channels': int64_feature(IMAGE_SHAPE[2]),
            'image/shape': int64_feature(IMAGE_SHAPE),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, dataset, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name, dataset)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, dataset, name):
    return '%s/%s_%s.tfrecord' % (output_dir, name, dataset)


def run(dataset_dir, output_dir, name='bib', dataset='training'):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    tf_filename = _get_output_filename(output_dir, dataset, name)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # Dataset filenames
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS, dataset)
    filenames = [fn.split('.')[0] for fn in os.listdir(path)]
    #filenames = sorted(os.listdir(path))

    # Process dataset files.
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, filename in enumerate(filenames):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
            sys.stdout.flush()

            _add_to_tfrecord(dataset_dir, dataset, filename, tfrecord_writer)

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the bib dataset!')
