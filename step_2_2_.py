import cv2
import numpy as np
import tensorflow as tf
import os
import time
from step_2_2_problem import *


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
    factor_size = 0.769
    start_time = time.time()
    while True:
        # capture image
        ret,raw_img = cap.read()
        if raw_img is None:
            break
        raw_img = cv2.resize(raw_img,(tWidth,tHeight))
        # for i in range(5):
        #     cap.grab()
        iloop = fps / 6 #Process x frames per second
        while iloop:
            cap.grab () #Only take frames without decoding,
            iloop =iloop - 1
            if iloop <1 :
                break 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
        # overlay with line
        distance = 45
        pt1 = ( int(sqsize/2-distance), 0 )
        pt2 = ( int(sqsize/2-distance), int(sqsize) )
        cv2.line(img, pt1, pt2, (0,0,255), 2)
        pt1 = ( int(sqsize/2+distance), 0 ) 
        pt2 = ( int(sqsize/2+distance), int(sqsize) )
        cv2.line(img, pt1, pt2, (0,0,255), 2)

        if class_id.size != 0:
            xmin = bbox[0][1]*tWidth
            xmax = bbox[0][3]*tWidth
            ymin = bbox[0][0]*tHeight
            ymax = bbox[0][2]*tHeight     
            if xmin < int(sqsize/2+distance) and not Entry and xmin > int(sqsize/2-distance):
                # print('in')
                Entry = True
            if xmin < int(sqsize/2-distance) and Entry:
                # print('out')
                Entry = False
                if class_id[0] == 1:
                    categoryList["lime"]["count"] = categoryList["lime"]["count"] + 1
                    w = (xmax - xmin)*factor_size
                    h = (ymax - ymin)*factor_size
                    categoryList["lime"]["size"].append(w*h)
                    print('lime detected' , categoryList["lime"]["count"] , "size:" , w*h)
                elif class_id[0] == 2:
                    categoryList["marker"]["count"] = categoryList["marker"]["count"] + 1
                    w = (xmax - xmin)*factor_size
                    h = (ymax - ymin)*factor_size
                    categoryList["marker"]["size"].append(w*h)
                    print('marker detected' , categoryList["marker"]["count"] , "size:" , w*h)
        # preview image
        cv2.imshow('Preview', img)     
        key = cv2.waitKey(int(1000/fps))
        if key == ord('q'):
            break
    cap.release()
    end_time = time.time()
    print("time used : " , end_time - start_time)
    print_result(categoryList)