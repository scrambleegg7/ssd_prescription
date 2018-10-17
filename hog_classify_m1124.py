import cv2
from cv2 import imshow
import numpy as np

import time
from builtins import FileExistsError
import logging, sys, os

from sklearn.externals import joblib
from hog_feature_extract import image_resize, color_hist, bin_spatial, get_hog_features, slide_window, draw_boxes
from hog_feature_extract import search_windows



#from ssdpipelineClass import SSDPipeline

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
    # setting parameters
    #
    mfilename = "./models/hog_svc_jlib.sav"
    logging.info("Loading svc model from file %s" % mfilename)
    svc =  joblib.load(mfilename)

    sfilename = "./models/xscaler_jlib.sav"    
    logging.info("Loading X_scaler from file %s" % sfilename)
    X_scaler = joblib.load(sfilename)

    color_space="YCrCb"
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)
    hist_bins = 32
    hist_range=(0,256)
    spatial_feat = True
    hist_feat = True
    hog_feat = True    

    #
    # open SSDPipeline class
    #
    parentDir = "./results"

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
        
        w = slide_window(frame, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(400,720), xy_overlap=(0.90, 0.90))

        hot_windows = search_windows(frame, w, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

        new_frame = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=6)                    

        if len(hot_windows) > 0:

            timestr = time.strftime("%Y%m%d-%H%M%S")
            MMDDHH = time.strftime("%m%d%H")

            subdir = os.path.join( parentDir, MMDDHH)
            try:
                os.makedirs(subdir)
            except FileExistsError as e:
                pass

            _num = cv2.countNonZero(thresh)
            if _num > 8000:
                logging.info("frame diffs happened bigger than threshhold --> %d" % _num )
                #logging.info("Name %s" % (classes,) )
                #logging.info("Probability %s" % (probs,)   )
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