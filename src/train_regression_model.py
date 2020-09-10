import argparse
import tensorflow as tf
import pathlib
import PIL
import numpy as np

import pathlib



parser = argparse.ArgumentParser(description="Train a model to predict number of people in an image")

parser.add_argument("--dataset", default="crowdhuman", 
                    help="Name of dataset. Subdirectory of data/regression_tfrecord")
parser.add_argument("--input-size", default=224, type=int, 
                    help="Side length of input. Image will be resized to NxN in preprocessing")
parser.add_argument("--model-prefix", default="mobilenet",
                    help="Prefix to be used in naming the model")
parser.add_argument("--alpha", default=0.125,
                    help="Depth multiplier. The smaller it is, the smaller the resulting model.")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", default=512)

def _example_to_tensors(example, input_size):
    example = tf.io.parse_example(
        example[tf.newaxis], {
            'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'image/num_people': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        })
    img_tensor =  tf.io.decode_jpeg(example['image/encoded'][0])
    img_tensor = tf.image.resize(img_tensor, (input_size, input_size))
    num_tensor = example['image/num_people'][0]
    return img_tensor, num_tensor

def load_dataset(dataset_name, input_size, split="train"):
    """
    split: 'train', 'val', or 'testdev'
    """
    datadir = pathlib.Path('data/regression_tfrecord') / dataset_name
    filenames = [str(p) for p in datadir.glob(f"coco_{split}.record*")]
    tfrecords = tf.data.TFRecordDataset(filenames)
    def _map_fn(example):
        return _example_to_tensors(example, input_size)
    dataset = tfrecords.map(_map_fn)
    return dataset
    
def build_model(input_size):
    input_shape = (input_size, input_size, 3)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    backbone = tf.keras.applications.MobileNet(
        input_shape = input_shape, alpha=0.1, include_top=False, weights=None
    )
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    regressor = tf.keras.Sequential(
        [tf.keras.layers.Conv2D(kernel_size=input_size//32, filters=32),
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
    return f"{args.model_prefix}_{args.alpha}_{args.input_size}_{args.dataset}"

def main():
    args = parser.parse_args()
    
    CKPT_PATH = f'models/{get_model_name(args)}/best_val.ckpt'
    
    model = build_model(args.input_size)
    if pathlib.Path(CKPT_PATH).exists():
        print("Loading checkpoint")
        model.load_weights(CKPT_PATH)
    
    train_dataset = load_dataset(args.dataset, args.input_size, split="train").batch(args.batch_size)
    val_dataset = load_dataset(args.dataset, args.input_size, split="val").batch(args.batch_size)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-2,
        decay_steps=150,
        decay_rate=0.96,
        staircase=True)
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(1.0),
        metrics=[tf.keras.losses.MeanAbsoluteError()])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f'models/{get_model_name(args)}/best_val.ckpt', save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4),
        tf.keras.callbacks.TensorBoard(log_dir=f'models/{get_model_name(args)}/tensorboard_logs')
    ]
    
    history = model.fit(x = train_dataset, validation_data = val_dataset, epochs=args.epochs, callbacks=callbacks)
    
    print(f"Model name: {get_model_name(args)}")
    model.save(f'exported_models/{get_model_name(args)}/saved_model')

    
if __name__ == "__main__":
    main()
    
    
    
        