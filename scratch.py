import tensorflow as tf
import pathlib

data_dir = "data/tfrecord/crowdhuman_debug"

def get_filenames(datadir: str):
    datadirpath = pathlib.Path(datadir)
    filenames = [str(p) for p in datadirpath.iterdir() if
            (not p.is_dir() and \
            p.stem == "coco_train")]
    return filenames

def get_dataset(data_dir):
    filenames = get_filenames(data_dir) 
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset

dataset = get_dataset(data_dir)

print(dataset)

for raw_record in dataset.take(1):  
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(repr(example))

