import cv2
import os
import sys
import glob
import numpy as np 
import pandas as pd

import time
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


import logging
from hog_feature_extract import image_resize, color_hist, bin_spatial, get_hog_features, extract_features

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

def proc(targetDir):

    #filename = os.path.join(targetDir , "Annotations")
    #img_dir = os.path.join(targetDir , "JPEGImages" )
    cut_image = os.path.join(targetDir , "cut_images")

    logging.info("Target File Directory %s " %   cut_image)
    ##filenames = os.listdir(cut_image)
    filenames = glob.glob( os.path.join(cut_image,"*.jpg") )
    logging.info("Total file counts -  %d" %  len(filenames))

    image_read_ops = lambda im : cv2.imread(im)
    bgr2rgb_ops = lambda im: cv2.cvtColor(im , cv2.COLOR_BGR2RGB)

    #gray_ops = lambda im: cv2.cvtColor(im , cv2.COLOR_RGB2GRAY)
    #YCrCb_ops = lambda im : cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)    
    image_resize_ops = lambda im : image_resize(im, size=(128,128))

    # skip ZERO size images
    images_list = [image_read_ops(f) for f in filenames if not image_read_ops(f) is None ]

    resized_images_list = list( map( image_resize_ops, images_list  ) )
    rgb_images_list = list( map( bgr2rgb_ops, resized_images_list  )  )

    logging.info( np.array( rgb_images_list )[0].shape  )
    return rgb_images_list

def feature_extraction(images_list):

    color_space="YCrCb"
    orient = 8
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)
    hist_bins = 32
    hist_range=(0,256)
    spatial_feat = True
    color_feat = True
    hog_feat = True    


    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')


    image_features = extract_features(images_list, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, 
                                    hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    color_feat=color_feat, hog_feat=hog_feat)


    logging.info(  np.array( image_features ).shape  )

    return image_features

## example on how to use it

def main(argv):

    if len(sys.argv) == 1:
        logging.info( 'Usage: # python %s DIRname (eg. VOC2007 or myVOC)' % sys.argv )
        #quit()                 

    logging.info("Extract Prescription features..")
    targetDir = os.path.join(  './VOCdevkit',  "myVOC" )
    resized_images_list = proc(targetDir)
    prescription_features = feature_extraction(resized_images_list)

    logging.info("Extract Bills features..")
    targetDir = os.path.join(  './VOCdevkit',  "Bills_output" )
    resized_images_list = proc(targetDir)
    bills_features = feature_extraction(resized_images_list)


    # y 
    y = np.hstack((np.ones(len(prescription_features)), 
              np.zeros(len(bills_features))))

    # X 
    X = np.vstack((prescription_features, bills_features)).astype(np.float64)                        
    # Fit a per-column scaler

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)

    #X_scaler = StandardScaler().fit(X)
    #X = X_scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    # Apply the scaler to X
    X_scaler = StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    #svc = LinearSVC()

    svc = SVC(kernel='linear', probability=True)

    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    
    t=time.time()
    print("Save model to file")
    mfilename = "./models/hog_svc_jlib.sav"
    joblib.dump(svc, mfilename)
    print( (time.time() - t ) , " Seconds to save SVC model....  ")

    sfilename = "./models/xscaler_jlib.sav"
    joblib.dump(X_scaler, sfilename)

if __name__ == "__main__":
    main(sys.argv[1:])

