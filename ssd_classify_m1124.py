import cv2
from cv2 import imshow
import numpy as np

import time
from builtins import FileExistsError
import logging, sys, os

from ssdpipelineClass import SSDPipeline

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def videopipeline():

    # AXIS M1124 Video streaming

    cam = cv2.VideoCapture()
    cam.open("http://192.168.1.151/axis-cgi/mjpg/video.cgi?fps=1")

    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)

    #
    # open SSDPipeline class
    #
    ssdp = SSDPipeline()
    ssdp.setClassColors()

    parentDir = "./results"

    prv_frame = np.zeros( (720,1280,3) )
    while(True):

        ret, frame = cam.read()

        new_frame, classes, probs = ssdp.pipeline(frame)

        if len(classes) > 0:
            logging.info("Name %s" % (classes,) )
            logging.info("Probability %s" % (probs,)   )

            if "Prescription" in classes:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                MMDDHH = time.strftime("%m%d%H")

                subdir = os.path.join( parentDir, MMDDHH)
                try:
                    os.makedirs(subdir)
                except FileExistsError as e:
                    pass


                print(  prv_frame.ravel().sum() )
                print(  frame.ravel().sum()  )

                if prv_frame.ravel().sum() != frame.ravel().sum() : 

                    cv2.imwrite(  os.path.join( subdir, 'prescription-%s.jpg' % timestr )    , frame )
                
        prv_frame = frame

        imshow("Source video", new_frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     




def main():
    
    videopipeline()



if __name__ == "__main__":
    main()