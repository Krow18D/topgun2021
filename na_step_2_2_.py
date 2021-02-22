import cv2
import numpy as np
import tensorflow as tf
import os
from step_2_2_problem import *


if __name__ == '__main__':
    cap = cv2.VideoCapture('clips/test_clip.h264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    sqsize = max(width,height)

    Entry = False
    count = 0
    factor_size = 0.274

    while True:
        # capture image
        ret,raw_img = cap.read()


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
            offset = int( (width - height)/2 )
            frame[offset:height+offset,:] = raw_img
        else:
            offset = int( (height - width)/2 )
            frame[:,offset:] = raw_img
        # problems
        class_id, bbox = detect_objects(frame)

        img = overlay_objects(frame)

        # overlay with line
        pt1 = ( int(sqsize/2-100), 0 )
        pt2 = ( int(sqsize/2-100), int(sqsize) )
        cv2.line(img, pt1, pt2, (0,0,255), 2)
        pt1 = ( int(sqsize/2+100), 0 )
        pt2 = ( int(sqsize/2+100), int(sqsize) )
        cv2.line(img, pt1, pt2, (0,0,255), 2)  

        if class_id.size != 0:
            # print(int(sqsize/2+100)/(sqsize))
            # print(bbox)
            if class_id[0] == 1:
                xmin = bbox[0][1]*width
                xmax = bbox[0][3]*width
                ymin = bbox[0][0]*height
                ymax = bbox[0][2]*height                
                if xmin < int(sqsize/2+100) and not Entry and xmin > int(sqsize/2-100):
                    Entry = True
                if xmax < int(sqsize/2+100) and Entry:
                    Entry = False
                    count = count + 1
                    w = (xmax - xmin)*factor_size
                    h = (ymax - ymin)*factor_size
                    lime_BBsize = w*h
                    print(count)
                    print('size :' + str(lime_BBsize) + ' mm^2')

        cv2.imshow('Preview', img)     
        key = cv2.waitKey(int(1000/fps))
        if key == ord('q'):
            break
    cap.release()
    print_result()