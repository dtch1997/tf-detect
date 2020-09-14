# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Convert raw COCO dataset to VisualWakeWords dataset for binary classification. 
TFRecords are labelled with whether they contain at least one person or not. 

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow.compat.v1 as tf

import sys
sys.path.append("lib/tfmodels/research")

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_boolean(
    'include_masks', False, 'Whether to include instance segmentations masks '
    '(PNG encoded) in the result. default: False.')
tf.flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')

tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file', '',
                       'Validation annotations JSON file.')


tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_integer('min_area', 96 * 48, 'Minimum area of bounding box. Smaller bounding boxes will be filtered out')


FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def clip_to_unit(x):
  return min(max(x, 0.0), 1.0)

def ann_is_valid(object_annotations, image_height, image_width, min_area):
    """
    object_annotations:
        dict with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] 

        Bounding box coordinates are to be given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner. 

    category_index:
        a dict containing COCO category information keyed by the
        'id' field of each category.  See the label_map_util.create_category_index
        function.
    """

    (x, y, width, height) = tuple(object_annotations['bbox'])
    if x < 0 or y < 0:
        return False
    if width <= 0 or height <= 0:
        return False
    if x + width > image_width or y + height > image_height:
        return False
    if width * height < min_area:
        return False
    return True

def ann_is_person(object_annotations, category_index):
    category_id = int(object_annotations['category_id'])
    category_name = category_index[category_id]['name'].encode('utf8')
    return category_name == b'person'

def create_tf_examples(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      min_area):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
        coordinates in the official COCO dataset are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed by the
      'id' field of each category.  See the label_map_util.create_category_index
      function.
    min_area: The minimum size of bounding box to keep. Smaller bounding boxes will be filtered out.

  Returns:
    keys: SHA256 hash of the cropped images.
    examples: A list of converted [tf.Example]
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
    
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  # Read in the image
  full_path = os.path.join(image_dir, filename)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    image = PIL.Image.open(fid)
    
  key = []
  example = []

  contains_person = False
  num_annotations_skipped = 0
    
    
  # If the image contains at least one person, it is a positive example. ]
  # Otherwise, it is a negative example. 
  for i, object_annotations in enumerate(annotations_list):
    if ann_is_valid(object_annotations, image_height, image_width, min_area) and ann_is_person(object_annotations, category_index):  
      contains_person = True
        
    else:
      num_annotations_skipped += 1    

  label = 1 if contains_person else 0
  with io.BytesIO() as output:
    image.save(output, format='jpeg')
    encoded_jpg = output.getvalue()
    key = hashlib.sha256(encoded_jpg).hexdigest()

  feature_dict = {
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/class':
          dataset_util.int64_feature(label)
  }
    
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return (key, example, num_annotations_skipped, label)


def _create_tf_record_from_coco_annotations(annotations_file, image_dir,
                                            output_path, min_area,
                                            num_shards):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
    keypoint_annotations_file: JSON file containing the person keypoint
      annotations. If empty, then no person keypoint annotations will be
      generated.
    densepose_annotations_file: JSON file containing the DensePose annotations.
      If empty, then no DensePose annotations will be generated.
    remove_non_person_annotations: Whether to remove any annotations that are
      not the "person" class.
    remove_non_person_images: Whether to remove any images that do not contain
      at least one "person" annotation.
  """
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = label_map_util.create_category_index(
        groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      logging.info('Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    logging.info('%d images are missing annotations.',
                 missing_annotation_count)

    total_num_annotations_skipped = 0
    total_positive = 0
    total_negative = 0
    
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]
      (_, tf_example, num_annotations_skipped, label) = create_tf_examples(
           image, annotations_list, image_dir, category_index, min_area)
      total_num_annotations_skipped += num_annotations_skipped
      
      if tf_example:
        shard_idx = idx % num_shards
        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        if label == 0:
          total_negative += 1
        else:
          total_positive += 1
    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)
    logging.info(f'Total of {total_positive} positie examples and {total_negative} negative examples created')


def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')

  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.min_area,
      num_shards=100)
  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_image_dir,
      val_output_path,
      FLAGS.min_area,
      num_shards=50)

if __name__ == '__main__':
  tf.app.run()
