import time
import os
import glob
import cv2
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


pipeline_path = "output/pipeline.config"
checkpoint_path = "output/checkpoints"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(checkpoint_path, 'ckpt-1')).expect_partial()

@tf.function
def detect(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

elapsed_time = []
images_np = []
for img_file in glob.glob('test/*.jpg'):
    img = cv2.imread(img_file)
    start_time = time.time()
    input_tensor = tf.convert_to_tensor([img], dtype=tf.float32)
    results = detect(input_tensor)
    end_time = time.time()
    elapsed_time.append( (end_time - start_time) )
    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()
    print(img_file, classes[scores > 0.8], bboxes[scores > 0.8])
print(elapsed_time)