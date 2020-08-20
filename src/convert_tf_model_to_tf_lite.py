import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import numpy as np
import pathlib

from PIL import Image

parser = argparse.ArgumentParser(description="Convert a TF SavedModel to a TFLite model")
parser.add_argument("--model-name", help="Name of the model. See tools/train_model.sh for semantics of model name")
parser.add_argument("--dataset", help="Name of the TFRecord dataset that should be used for quantization", default="crowdhuman_debug") 
parser.add_argument("--num-samples", help="Number of samples to calibrate on", type=int, default=100)

def fake_data_gen(num_samples):
    def representative_dataset_gen():
        for i in range(num_samples):
            yield [np.ones((1, 96, 96, 3)).astype(np.float32)]
    return representative_dataset_gen

def make_data_gen(dataset_name, num_samples):
    """
    Uses the images from datadir/Images to quantize the model. 
    """
    if dataset_name == "fake":
        return fake_data_gen(num_samples)

    datadir = pathlib.Path('data/raw') / dataset_name
    imgdir = datadir / 'Images'
    
    def representative_dataset_gen():
        for i, filename in enumerate(imgdir.iterdir()):
            if filename.suffix not in ['.jpeg', '.jpg', '.png']: 
                continue
            image = Image.open(str(filename.resolve()))
            image = image.resize((96, 96))
            yield [np.array(image).reshape(-1, 96, 96, 3)]
            if i >= num_samples: break
    return representative_dataset_gen

def main():
    args = parser.parse_args()
    model_savedir = f'exported_models/{args.model_name}/saved_model'

    converter = tf.lite.TFLiteConverter.from_saved_model(model_savedir, signature_keys=['serving_default'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_data_gen(args.dataset, args.num_samples)
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized_model = converter.convert()
    bytes = open(f'exported_models/{args.model_name}/model.tflite', "wb").write(quantized_model)

    
if __name__ == "__main__":
    main()
