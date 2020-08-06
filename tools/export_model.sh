#!/bin/bash

USAGE="source tools/train_model.sh MODEL_NAME"
PIPELINE_CONFIG_PATH="configs/${1}"
MODEL_DIR="models/${1}"
EXPORT_DIR="exported_models/${1}"
echo "Usage: ${USAGE}"

python lib/tfmodels/research/object_detection/exporter_main_v2.py \
	--input_type image_tensor \
	--pipeline_config_path $PIPELINE_CONFIG_PATH \
       	--trained_checkpoint_dir $MODEL_DIR \
	--output_directory $EXPORT_DIR
