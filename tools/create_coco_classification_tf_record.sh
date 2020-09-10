#!/bin/bash

USAGE="source tools/create_coco_tf_record.sh DATASET_NAME"
echo "Expected usage: ${USAGE}"
echo "Using ${1} as DATASET_NAME..."

# Script expects the following directory structure:

# data/raw/DATASET_NAME
# - Images 		(directory of training images)
# - Images_val 		(directory of val images)
# - annotations		(directory of annotations in COCO format)
#   - train.json 	(train annotations)
#   - val.json		(val annotations)

DATASET_NAME=$1

DATA_DIR="data/raw/${DATASET_NAME}"
TRAIN_IMAGE_DIR="${DATA_DIR}/Images"
VAL_IMAGE_DIR="${DATA_DIR}/Images_val"
TEST_IMAGE_DIR="${DATA_DIR}/Images_val"
TRAIN_ANNOTATIONS_FILE="${DATA_DIR}/annotations/train.json"
VAL_ANNOTATIONS_FILE="${DATA_DIR}/annotations/val.json"
TESTDEV_ANNOTATIONS_FILE="${DATA_DIR}/annotations/val.json"
OUTPUT_DIR="data/classification_tfrecord/${DATASET_NAME}"
python src/create_coco_classification_tf_record.py --logtostderr \
	--train_image_dir=$TRAIN_IMAGE_DIR \
	--val_image_dir=$VAL_IMAGE_DIR \
	--test_image_dir=$TEST_IMAGE_DIR \
	--train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
	--val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
	--testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
	--output_dir="${OUTPUT_DIR}"
