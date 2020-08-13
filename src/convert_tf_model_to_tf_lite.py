import tensorflow as tf
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Convert a TF SavedModel to a TFLite model")
parser.add_argument("--model-name", help="Name of the model. See tools/train_model.sh for semantics of model name")
parser.add_argument("--dataset", help="Name of the TFRecord dataset that should be used for quantization") 
parser.add_argument("--num-samples", help="Number of samples to calibrate on")

args = parser.parse_args()
model_savedir = f'exported_models/{args.model_name}/saved_model'
data_dir = f'data/tfrecord/{args.dataset}'

def get_filenames(datadir: str):
    datadirpath = pathlib.Path(datadir)
    filenames = [p for p in datadirpath.iterdir() if
            (not p.is_dir() and \
             p.suffix == ".record")]
    return filenames

def representative_dataset_gen():
    dataset = tf.data.TFRecordDataset(get_filenames(data_dir))
    for raw_record in dataset.take(args.num_samples):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())


representative_dataset_gen()

converter = tf.lite.TFLiteConverter.from_saved_model(model_savedir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
quantized_model = converter.convert()
bytes = open(f'exported_models/{args.model_name}/model.tflite', "wb").write(quantized_model)

print(f"Model of {bytes} bytes was written")
