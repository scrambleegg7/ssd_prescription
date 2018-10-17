import numpy as np  
import cv2


import os
import sys
import glob
import logging
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.externals import joblib

from hog_feature_extract import image_resize, color_hist, bin_spatial, get_hog_features, slide_window, draw_boxes
from hog_feature_extract import search_windows

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)



def testcase1():

    targetDir = os.path.join(  './VOCdevkit',  "Bills_output" )
    image_dir = os.path.join( targetDir, "JPEGImages" )

    logging.info("Image File Directory %s " %   image_dir)
    filenames = glob.glob( os.path.join(image_dir,"*.jpg") )
    logging.info("Total file counts -  %d" %  len(filenames))

    image_read_ops = lambda im : cv2.imread(im)
    bgr2rgb_ops = lambda im: cv2.cvtColor(im , cv2.COLOR_BGR2RGB)
    image_resize_ops = lambda im : image_resize(im, size=(128,128))

    # skip ZERO size images
    images_list = [image_read_ops(f) for f in filenames if not image_read_ops(f) is None ]
    rgb_images_list = list( map( bgr2rgb_ops, images_list  )  )

    i = np.random.randint( len(filenames) )
    test_image = rgb_images_list[i]

    logging.info("main screen size %s" % (test_image.shape,)  )
    w = slide_window(test_image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(400,720), xy_overlap=(0.90, 0.90))

    logging.info("splitted windows numbers : %d" % len(w)  )
    #logging.info( w )

    #box_w = draw_boxes( test_image, w )
    #plt.imshow(box_w)
    #plt.show()
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

    hot_windows = search_windows(test_image, w, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(test_image, hot_windows, color=(0, 0, 255), thick=6)                    

    plt.imshow(window_img)
    plt.show()

def main():

    testcase1()

if __name__ == "__main__":
    main()