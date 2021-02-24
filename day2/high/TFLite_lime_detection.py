######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import time
# import requests
import sys
import importlib.util
from statistics import mean

from apiFire import *

def print_result(categoryList):
    print('Finished')
    print('found lime : ' , categoryList["lime"]["count"] , 'times')
    print('all limes size : ', categoryList["lime"]["size"])
    print('found marker : ' , categoryList["marker"]["count"] , 'times')
    print('all markers size : ', categoryList["marker"]["size"])



# def overlay_objects(img):
#     # img = img.copy()
#     # img = cv2.imread(img_file)
#     imgNp = tf.convert_to_tensor([img], dtype=tf.float32)
#     results = detect(imgNp)
#     bboxes = results['detection_boxes'][0].numpy()
#     classes = results['detection_classes'][0].numpy().astype(np.uint32) + 1
#     scores = results['detection_scores'][0].numpy()
#     if classes != [] and bboxes != []:
#         plotimg = plot_detections(img,bboxes,classes,scores,category_index)
#         return plotimg
#     return img.copy()


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.75)
parser.add_argument('--video', help='Name of the video file',
                    default='clips/test_clip.h264')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME != 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)
# print(VIDEO_PATH)
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
# sqsize = max(imW,imH)
sqsize = 320
# trainSize = 320
# tWidth = sqsize
# tHeight = int(imH * (tWidth / imW))

Entry = False
factor_size = 0.769
imgName = "limecapture.jpg"

categoryList = {
    "lime" : {"count":0,"size":[]},
    "marker" : {"count":0,"size":[]}
}

distance = 80
start = time.time()
elapsed_time = []
while(video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_rgb = frame
    frame_resized = cv2.resize(frame_rgb, (width,height))
    #frame_resized = cv2.resize(frame_rgb, (480, 320))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    start_elapsetime = time.time()
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    end_elapsetime = time.time()
    elapsed_time.append( (end_elapsetime - start_elapsetime) )
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # overlay with line

    pt1 = ( int(2*imW/3-distance), 0 )
    pt2 = ( int(2*imW/3-distance), int(imH) )
    cv2.line(frame, pt1, pt2, (0,0,255), 2)
    pt1 = ( int(2*imW/3+distance), 0 ) 
    pt2 = ( int(2*imW/3+distance), int(imH) )
    cv2.line(frame, pt1, pt2, (0,0,255), 2)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text



            # if class_id.size != 0:    
            if xmin < int(2*imW/3+distance) and not Entry and xmin > int(2*imW/3-distance):
                # print('in')
                Entry = True
            if xmin < int(2*imW/3-distance) and Entry:
                # print('out')
                Entry = False
                if int(classes[i]) == 0:
                    t = time.time()
                    smallSize = 15000
                    imgName = "limecapture.jpg"
                    categoryList["lime"]["count"] = categoryList["lime"]["count"] + 1
                    w = (xmax - xmin)*factor_size
                    h = (ymax - ymin)*factor_size
                    categoryList["lime"]["size"].append(w*h)
                    print(f'{t - start : .2f}' , 's : lime detected' , categoryList["lime"]["count"] , "size:" , w*h, " mm^2")
                    # limeFlag = True
                    if categoryList["lime"]["size"][categoryList["lime"]["count"] - 1] < smallSize:
                        cv2.imwrite(imgName, frame)
                        detectSend("Lime","S",imgName)
                    else:
                        detectSend("Lime","L","")
                elif int(classes[i]) == 1:
                    t = time.time()
                    categoryList["marker"]["count"] = categoryList["marker"]["count"] + 1
                    w = (xmax - xmin)*factor_size
                    h = (ymax - ymin)*factor_size
                    categoryList["marker"]["size"].append(w*h)
                    print(f'{t - start : .2f}' , 's : marker detected' , categoryList["marker"]["count"] , "size:" , w*h, " mm^2")
                    detectSend("Marker","L","")
    
    # All the results have been drawn on the frame, so it's time to display it.
    frame = cv2.resize(frame, (480, 320))
        
    cv2.imshow('TFLITE Lime detector', frame)
    limeFlag = False
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    
end_time = time.time()
# Clean up
video.release()
print_result(categoryList)
print('[Program] Usage time : ' , end_time-start , ' s')
print('[Object] Average Detection time : ' , mean(elapsed_time) , ' s')
cv2.destroyAllWindows()


