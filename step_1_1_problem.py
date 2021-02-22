import cv2
import numpy as np
import os

print(cv2.__version__)

if __name__ == '__main__':
    # 1. scan clips
    clipPath = './clips/'
    clips = []
    clist = os.listdir(clipPath)
    for c in clist:
        if c.endswith('.h264'):
            clips.append(clipPath + c)
    # 2. preview and capture images
    idx = 0
    imgPath = './image/'
    running = True
    for clip in clips:
        print('clip: ' + clip)
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
        height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
        sqsize = max(width,height)
        print("FPS: " + str(fps))
        ret,frame = cap.read()
        print("Image size: " + str(frame.shape))
        # 2.2 preview clip         
        while True:
            # 2.2.1 read image            
            ret,raw_img = cap.read()     
            if not ret:
                break
            frame = np.zeros((sqsize,sqsize,3),np.uint8)
            if width > height:
                offset = int((width-height)/2)
                frame[offset:height+offset,:] = raw_img
            else:
                offset = int((height-width)/2)
                frame[:,offset:] = raw_img
            cv2.imshow('Preview', frame)        
            key = cv2.waitKey(int(1000/fps))
            # 2.2.3 process commands
            if key == ord('n'):
                break
            if key == ord('q'): # Press q for quit
                running = False
                break
            if key == 32:
                fname = imgPath + 'lime_' + f'{idx:03}' + '.jpg'
                print('saving to')
                cv2.imwrite(fname,frame)
                idx=idx+1
        cap.release()
        if running == False:
            break
