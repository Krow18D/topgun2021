import cv2
import numpy as np
import tensorflow as tf
import os
import time
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

category_index = {1: {'id': 1, 'name': 'lime'}, 2: {'id': 2, 'name': 'marker'},}

@tf.function
def detect(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index):
    image_np_with_annotations = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    return image_np_with_annotations

def detect_objects(img):
    # class_id = 1
    # bbox = [0,0,0,0]
    # img = cv2.imread(img)
    imgNp = tf.convert_to_tensor([img], dtype=tf.float32)
    results = detect(imgNp)
    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()
    return (classes[scores > 0.8], bboxes[scores > 0.8])


def overlay_objects(img):
    # img = img.copy()
    # img = cv2.imread(img_file)
    imgNp = tf.convert_to_tensor([img], dtype=tf.float32)
    results = detect(imgNp)
    bboxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
    scores = results['detection_scores'][0].numpy()
    if classes != [] and bboxes != []:
        plotimg = plot_detections(img,bboxes,classes,scores,category_index)
        return plotimg
    return img.copy()

if __name__ == '__main__':
    cap = cv2.VideoCapture('clips/test_clip.h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    sqsize = 320
    # trainSize = 320
    tWidth = sqsize
    tHeight = int(height * (tWidth / width))
    categoryList = {
        "lime" : {"count":0,"size":[]},
        "marker" : {"count":0,"size":[]}
    }
    Entry = False
    start_time = time.time()
    while True:
        # capture image
        ret,raw_img = cap.read()
        if raw_img is None:
            break
        raw_img = cv2.resize(raw_img,(tWidth,tHeight))

        if not ret:
            break
        # add margin
        frame = np.zeros((sqsize,sqsize,3), np.uint8)
        if width > height:
            offset = int( (tWidth - tHeight)/2 )
            frame[offset:tHeight+offset,:] = raw_img
        else:
            offset = int( (tHeight - tWidth)/2 )
            frame[:,offset:] = raw_img
        # problems
        class_id, bbox = detect_objects(frame)
        img = overlay_objects(frame)

        # preview image
        cv2.imshow('Preview', img)     
        key = cv2.waitKey(int(1000/fps))
        if key == ord('q'):
            break
    end_time = time.time()
    cap.release()