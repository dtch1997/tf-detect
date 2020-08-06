import tensorflow as tf

saved_model_dir = "/home/dtch009/tf-detect/exported_models/ssd_mobilenet_v2_96x96_depth=0.2/saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
quantized_model = converter.convert()

