#!/bin/bash
# Install Object Detection API

# Clone source
git submodule update --init --recursive
cd lib/tfmodels/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install from source
cp object_detection/packages/tf2/setup.py .
python -m pip install .
# Run tests
python object_detection/builders/model_builder_tf2_test.py
