import cv2
from cv2 import imshow

import matplotlib.pyplot as plt

import numpy as np
import pickle

if __name__ == '__main__':
    print('Hello World')
    
    # Connect to video Source
    cam = cv2.VideoCapture()
    #
    #
    #
    # AXIS M1124 Video streaming
    cam.open("http://192.168.1.151/axis-cgi/mjpg/video.cgi?fps=1")

    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)
    
    BoardSize = (9,6)
        
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((BoardSize[0]*BoardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:BoardSize[0],0:BoardSize[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    i = 0
    
    while(True):
        ret, frame = cam.read()                      
             
        imshow("Source video", frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     
