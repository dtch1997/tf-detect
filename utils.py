from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection.builders import model_builder
from object_detection import model_lib
from object_detection import inputs
MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP

def load_configs_from_path(config_path):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        'get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
        'merge_external_params_with_configs']
    configs = get_configs_from_pipeline_file(
        config_path)
    configs = merge_external_params_with_configs(
        configs, None)
    return configs

def build_model(model_config):
    return model_builder.build(
        model_config=model_config, is_training=True)

def load_checkpoint(model, checkpoint_dir):
    global_step = tf.compat.v2.Variable(
        0, trainable=False, dtype=tf.compat.v2.dtypes.int64)
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=model)
    latest_checkpoint=tf.train.latest_checkpoint(
        checkpoint_dir, latest_filename=None
    )
    ckpt.restore(latest_checkpoint).expect_partial()
    return model

def make_eval_inputs(model, configs):
    return inputs.eval_input(eval_config = configs['eval_config'], 
                             eval_input_config = configs['eval_input_config'],
                             model_config = configs['model'],
                             model = model)

def run_inference(model, input_tensor):
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                  image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict