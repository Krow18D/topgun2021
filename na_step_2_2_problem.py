import cv2
import numpy as np
import tensorflow as tf


import os
import glob
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

pipeline_path = "output/pipeline.config"
checkpoint_path = "output/checkpoints"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(checkpoint_path, 'ckpt-1')).expect_partial()



def detect_objects(img):
    # class_id = 1
    # bbox = [0,0,0,0]
    input_tensor = tf.convert_to_tensor([img], dtype=tf.float32)
    input_tensor, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(input_tensor, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    bbox = detections['detection_boxes'][0].numpy()
    class_id = detections['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = detections['detection_scores'][0].numpy()

    return (class_id[scores > 0.8], bbox[scores > 0.8])


def overlay_objects(img):
    return img.copy()

def print_result():
    print('Finished')