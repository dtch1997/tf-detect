#!/bin/bash 
USAGE="source tools/train_model.sh MODEL_NAME"
PIPELINE_CONFIG_PATH="configs/${1}"
MODEL_DIR="models/${1}"
echo "Usage: ${USAGE}"

COMMAND="
cp ${PIPELINE_CONFIG_PATH} ${MODEL_DIR}/model.config &&
python lib/tfmodels/research/object_detection/model_main_tf2.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--model_dir=${MODEL_DIR} \
	--alsologtostderr"

echo "Executing ${COMMAND}"
eval $COMMAND
