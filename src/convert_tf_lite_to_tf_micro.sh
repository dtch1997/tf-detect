#!/bin/bash

USAGE="bash src/convert_tf_lite_to_tf_micro.sh MODEL_NAME"
MODEL_NAME=$1
EXPORT_DIR="exported_models/$MODEL_NAME"

xxd -i $EXPORT_DIR/model.tflite > $EXPORT_DIR/model_data.cc
