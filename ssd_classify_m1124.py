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
    cam.open("http://192.168.1.151/axis-cgi/mjpg/video.cgi?fps=4")

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

    parentDir = "/Volumes/ssd/FTP/"

    prv_frame = None
    while(True):

        ret, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prv_frame is None:
            prv_frame = gray
            continue

        frameDelta = cv2.absdiff(prv_frame, gray)
        # if frameDelta = difference less than 30, black,
        # frameDelta bigger than 30, then white
        thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
        
        new_frame, classes, probs = ssdp.pipeline(frame)

        if len(classes) > 0:

            if "Prescription" in classes:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                MMDDHH = time.strftime("%m%d%H")

                subdir = os.path.join( parentDir, MMDDHH)
                try:
                    os.makedirs(subdir)
                except FileExistsError as e:
                    pass

                _num = cv2.countNonZero(thresh)
                if _num > 6000:
                    logging.info("frame diffs happened bigger than threshhold --> %d" % _num )
                    logging.info("Name %s" % (classes,) )
                    logging.info("Probability %s" % (probs,)   )

                    classes = np.array(classes)
                    probs = np.array(probs)
                    prescri_probs = probs[classes == "Prescription"]
                    over80_probs = prescri_probs[ prescri_probs > 0.60 ]
                    if len(over80_probs) > 0:
                        logging.info("writing prescription image on disk."   )
                        cv2.imwrite(  os.path.join( subdir, 'prescription-%s.jpg' % timestr )    , frame )
                
        prv_frame = gray

        #imshow("frameDelta", frameDelta)
        #imshow("thresh", thresh)
        imshow("marked", new_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break     




def main():
    
    videopipeline()



if __name__ == "__main__":
    main()