# Visual Wake Words for Tensorflow 2.0

This repository contains Tensorflow 2.0 code to fine-tune a pretrained MobileNetV1 model on the Visual Wake Words dataset, and export the trained model to Tensorflow Lite for Microcontrollers. 

## Quick Start

Clone the repository:
```
git clone https://github.com/dtch1997/tf-detect.git
cd tf-detect
```
Download the COCO2014 dataset:
```
bash src/download_mscoco.sh data/raw/coco2014 2014
```
Train a model:
```
python src/train_vww_model.py
# The training script prints the model name to stdout. 
# E.g. 'Model name: vww_mobilenet_0.25_96_96_coco2014`
```

Convert the model to Tensorflow Lite:
```
python src/convert_tf_model_to_tf_lite.py --model-name MODEL_NAME 
```

Convert the model to Tensorflow Lite Micro:
```
bash src/convert_tf_lite_to_tf_micro.sh MODEL_NAME
```

## Custom training
To edit model hyperparameters, set the corresponding flags in `src/train_vww_model.py`:
```
python src/train_vww_model.py --help

usage: train_vww_model.py [-h] [--dataset DATASET]
                          [--input-height INPUT_HEIGHT]
                          [--input-width INPUT_WIDTH]
                          [--model-prefix MODEL_PREFIX] [--alpha ALPHA]
                          [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                          [--verbose VERBOSE] [--learning-rate LEARNING_RATE]
                          [--decay-rate DECAY_RATE] [--deploy]

Train a model to predict whether an image contains a person

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of dataset. Subdirectory of data/vww_tfrecord
  --input-height INPUT_HEIGHT
                        Height of input
  --input-width INPUT_WIDTH
                        Width of input
  --model-prefix MODEL_PREFIX
                        Prefix to be used in naming the model
  --alpha ALPHA         Depth multiplier. The smaller it is, the smaller the
                        resulting model.
  --epochs EPOCHS       Training procedure runs through the whole dataset once
                        per epoch.
  --batch-size BATCH_SIZE
                        Number of examples to process concurrently
  --verbose VERBOSE     Printing verbosity of Tensorflow model.fit()Set
                        --verbose=1 for per-batch progress bar, --verbose=2
                        for per-epoch
  --learning-rate LEARNING_RATE
                        Initial learning rate of SGD training
  --decay-rate DECAY_RATE
                        Number of steps to decay learning rate after
  --deploy              Set flag to skip training and simply export the
                        trained model
```

## Custom deployment
To edit conversion hyperparameters, set the corresponding flags in `src/convert_tf_model_to_tf_lite.py`:
```
usage: convert_tf_model_to_tf_lite.py [-h] [--model-name MODEL_NAME]
                                      [--dataset DATASET]
                                      [--num-samples NUM_SAMPLES]
                                      [--input-height INPUT_HEIGHT]
                                      [--input-width INPUT_WIDTH]

Convert a TF SavedModel to a TFLite model

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Name of the model. See tools/train_model.sh for
                        semantics of model name
  --dataset DATASET     Name of the TFRecord dataset that should be used for
                        quantization
  --num-samples NUM_SAMPLES
                        Number of samples to calibrate on
  --input-height INPUT_HEIGHT
  --input-width INPUT_WIDTH
```

## References: 
The Visual Wake Words dataset was introduced in (Chowdhery et al. 2019):
```
@article{DBLP:journals/corr/abs-1906-05721,
  author    = {Aakanksha Chowdhery and
               Pete Warden and
               Jonathon Shlens and
               Andrew Howard and
               Rocky Rhodes},
  title     = {Visual Wake Words Dataset},
  journal   = {CoRR},
  volume    = {abs/1906.05721},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.05721},
  archivePrefix = {arXiv},
  eprint    = {1906.05721},
  timestamp = {Mon, 24 Jun 2019 17:28:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-05721.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
