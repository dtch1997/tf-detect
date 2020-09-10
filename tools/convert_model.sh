#!/bin/bash

USAGE="source tools/train_model.sh MODEL_NAME INPUT_SIZE"
MODEL_NAME=${1}
EXPORT_DIR="exported_models/${1}"
INPUT_SIZE=${2}
echo "Usage: ${USAGE}"

python src/convert_tf_model_to_tf_lite.py \
	--model-name $MODEL_NAME \
	--dataset crowdhuman \
	--num-samples 15000 \
    --input-size $INPUT_SIZE

xxd -i "${EXPORT_DIR}/model.tflite" > "${EXPORT_DIR}/model_data.cc"
