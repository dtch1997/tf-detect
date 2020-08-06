#!/bin/bash 
USAGE="source tools/train_model.sh PIPELINE_CONFIG_PATH MODEL_DIR"
PIPELINE_CONFIG_PATH=$1
MODEL_DIR=$2
CHECKPOINT_DIR=${MODEL_DIR}
echo "Usage: ${USAGE}"
echo "Using PIPELINE_CONFIG_PATH=${PIPELINE_CONFIG_PATH}"
echo "Using MODEL_DIR=${MODEL_DIR}"

python lib/tfmodels/research/object_detection/model_main_tf2.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--model_dir=${MODEL_DIR} \
	--checkpoint_dir=${MODEL_DIR} \
	--alsologtostderr
