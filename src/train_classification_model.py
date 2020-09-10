import argparse
import tensorflow as tf
import pathlib
import PIL
import numpy as np

import pathlib
from collections import namedtuple

# Disable a lot of useless warnings
tf.get_logger().setLevel('WARNING')
ImageShape = namedtuple('ImageShape', 'height width channels')

parser = argparse.ArgumentParser(description="Train a model to predict whether an image is a person")

parser.add_argument("--dataset", default="crowdhuman", 
                    help="Name of dataset. Subdirectory of data/classification_tfrecord")
parser.add_argument("--input-height", default=64, type=int, 
                    help="Height of input")
parser.add_argument("--input-width", default=32, type=int, 
                    help="Width of input")
parser.add_argument("--model-prefix", default="mobilenet",
                    help="Prefix to be used in naming the model")
parser.add_argument("--alpha", type=float, default=0.125,
                    help="Depth multiplier. The smaller it is, the smaller the resulting model.")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", default=512)
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--decay-rate", type=float, default=0.98)

def _example_to_tensors(example, input_shape):
    """
    @brief: Read a serialized tf.train.Example and convert it to a (image, label) pair of tensors.
            TFRecords are created using src/create_coco_classification_tf_record.py 
    @author: Daniel Tan
    """
    example = tf.io.parse_example(
        example[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape = (), dtype=tf.string),
            'image/class': tf.io.FixedLenFeature(shape = (), dtype=tf.int64)
        })
    img_tensor =  tf.io.decode_jpeg(example['image/encoded'][0])
    img_tensor = tf.image.resize(img_tensor, size=(input_shape.height, input_shape.width))
    label = example['image/class']
    return img_tensor, label

def load_dataset(dataset_name, input_shape, split="train"):
    """
    Parameters: 
        split: 'train' or 'val'
        dataset_name: A subdirector of data/classification_tfrecord
        input_shape: An ImageShape instance
        
    Return: 
        A dataset where each entry is a (image, label) tuple
    """
    datadir = pathlib.Path('data/classification_tfrecord') / dataset_name
    filenames = [str(p) for p in datadir.glob(f"coco_{split}.record*")]
    tfrecords = tf.data.TFRecordDataset(filenames)
    def _map_fn(example):
        return _example_to_tensors(example, input_shape)
    dataset = tfrecords.map(_map_fn)
    return dataset
    
def build_model(input_shape, alpha):
    """
    Build a MobilenetV1 architecture with given input shape and alpha. 
    
    Parameters:
        input_shape: An ImageShape instance
        alpha: A float between 0 and 1. Model size scales with (alpha^2).
     
    Returns:
        A newly initialized model with the given architecture. 
    """
    input_shape = (input_shape.height, input_shape.width, input_shape.channels)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    ])
    backbone = tf.keras.applications.MobileNet(
        input_shape = input_shape, alpha=alpha, include_top=False, weights=None
    )
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    regressor = tf.keras.Sequential(
        [tf.keras.layers.GlobalAveragePooling2D(),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(1, activation=None)]
    )

    inputs = tf.keras.Input(input_shape)
    x = inputs
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = backbone(x)
    outputs = regressor(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def get_model_name(args):
    return f"classifier_{args.model_prefix}_{args.alpha}_{args.input_height}_{args.input_width}_{args.dataset}"

def get_checkpoint_dir(args):
    return f'models/{get_model_name(args)}/best_val.ckpt'

def get_model_dir(args):
    return f'exported_models/{get_model_name(args)}/saved_model'      

def main():
    args = parser.parse_args()
    input_shape = ImageShape(height=args.input_height, width=args.input_width, channels=3)
    CKPT_PATH = get_checkpoint_dir(args)
    
    model = build_model(input_shape, args.alpha)
    if pathlib.Path(CKPT_PATH).exists():
        print("Loading checkpoint")
        model.load_weights(CKPT_PATH)
    else:
        print("Model checkpoint not found, new model initialized")
    
    train_dataset = load_dataset(args.dataset, input_shape, split="train").shuffle(1024).batch(args.batch_size)
    val_dataset = load_dataset(args.dataset, input_shape, split="val").shuffle(1024).batch(args.batch_size)
    
    # Inspect input values
    """
    for image, label in train_dataset.shuffle(128).take(5):
        print(image, label)
    for image, label in val_dataset.shuffle(128).take(5):
        print(image, label)
    """
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate,
        decay_steps=150,
        decay_rate=args.decay_rate,
        staircase=True)
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CKPT_PATH, save_weights_only=True, monitor='accuracy', save_best_only=True, save_freq='epoch'),
        # tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=4)
    ]
    
    history = model.fit(x = train_dataset, validation_data = val_dataset, epochs=args.epochs, callbacks=callbacks, verbose=args.verbose)
    
    print(f"Model name: {get_model_name(args)}")
    model.save(get_model_dir(args))

    
if __name__ == "__main__":
    main()
    
    
    
        