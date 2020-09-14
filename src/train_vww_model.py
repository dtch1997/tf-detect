import argparse
import tensorflow as tf
import pathlib

from collections import namedtuple

# Disable a lot of useless warnings
tf.get_logger().setLevel('ERROR')
ImageShape = namedtuple('ImageShape', 'height width channels')

parser = argparse.ArgumentParser(description="Train a model to predict whether an image contains a person")

parser.add_argument("--dataset", default="coco2014", 
                    help="Name of dataset. Subdirectory of data/vww_tfrecord")
parser.add_argument("--input-height", default=96, type=int, 
                    help="Height of input")
parser.add_argument("--input-width", default=96, type=int, 
                    help="Width of input")
parser.add_argument("--model-prefix", default="mobilenet",
                    help="Prefix to be used in naming the model")
parser.add_argument("--alpha", type=float, default=0.25,
                    help="Depth multiplier. The smaller it is, the smaller the resulting model.")
parser.add_argument("--epochs", type=int, default=20, 
                    help="Training procedure runs through the whole dataset once per epoch.")
parser.add_argument("--batch-size", type=int, default=512,
                    help="Number of examples to process concurrently")
parser.add_argument("--verbose", type=int, default=2,
                    help="Printing verbosity of Tensorflow model.fit()"
                    "Set --verbose=1 for per-batch progress bar, --verbose=2 for per-epoch")
parser.add_argument("--learning-rate", type=float, default=1e-3,
                    help="Initial learning rate of SGD training")
parser.add_argument("--decay-rate", type=float, default=0.98,
                    help="Number of steps to decay learning rate after")

parser.add_argument("--deploy", action='store_true',
                    help="Set flag to skip training and simply export the trained model")

def _example_to_tensors(example, input_shape):
    """
    @brief: Read a serialized tf.train.Example and convert it to a (image, label) pair of tensors.
            TFRecords are created using src/create_coco_vww_tf_record.py 
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
        dataset_name: A subdirector of data/vww_tfrecord
        input_shape: An ImageShape instance
        
    Return: 
        A dataset where each entry is a (image, label) tuple
    """
    datadir = pathlib.Path('data/vww_tfrecord') / dataset_name
    filenames = [str(p) for p in datadir.glob(f"coco_{split}.record*")]
    tfrecords = tf.data.TFRecordDataset(filenames)
    def _map_fn(example):
        return _example_to_tensors(example, input_shape)
    dataset = tfrecords.map(_map_fn)
    return dataset.filter(lambda x, y: tf.shape(x)[2] == 3)
    
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
    backbone = tf.keras.applications.MobileNet(
        input_shape = input_shape, alpha=alpha, include_top=False, weights='imagenet'
    )
    classifier = tf.keras.Sequential(
        [tf.keras.layers.GlobalAveragePooling2D(),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(1, activation=None)]
    )

    inputs = tf.keras.Input(input_shape)
    x = inputs
    x = backbone(x)
    outputs = classifier(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def get_model_name(args):
    return f"vww_{args.model_prefix}_{args.alpha}_{args.input_height}_{args.input_width}_{args.dataset}"

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
        print("Previous checkpoint found; loading saved weights")
        model.load_weights(CKPT_PATH)
    
    if not args.deploy:
        train_dataset = load_dataset(args.dataset, input_shape, split="train").shuffle(1024).batch(args.batch_size)
        val_dataset = load_dataset(args.dataset, input_shape, split="val").shuffle(1024).batch(args.batch_size)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_steps=100000,
            decay_rate=args.decay_rate,
            staircase=True)
        
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                CKPT_PATH, 
                save_weights_only=True, 
                monitor='accuracy',
                mode = 'max',
                save_best_only=True, 
                save_freq='epoch')
        ]
        
        history = model.fit(x = train_dataset, validation_data = val_dataset, epochs=args.epochs, callbacks=callbacks, verbose=args.verbose)
        
    print(f"Model name: {get_model_name(args)}")
    model.save(get_model_dir(args))

    
if __name__ == "__main__":
    main()
    
    
    
        
