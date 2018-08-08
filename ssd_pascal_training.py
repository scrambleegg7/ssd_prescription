#
import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

# user-defined module
from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

from Generator import Generator

import logging, sys, os

base_lr = 3e-4

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

def train_proc(dirName="VOC2007"):

    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['image.interpolation'] = 'nearest'

    np.set_printoptions(suppress=True)

    # 21
    NUM_CLASSES = 21 #4
    input_shape = (300, 300, 3)

    priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # gt = pickle.load(open('gt_pascal.pkl', 'rb'))
    pascal_file = './PASCAL_VOC/%s.pkl' % dirName
    print("loading ", pascal_file)
    gt = pickle.load( open( pascal_file, 'rb') )
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    BATCH_SIZES = 4
    logging.info(" training number: %d  validation number: %d " % (num_train, num_val) )

    steps_per_epoch = num_train / BATCH_SIZES
    validation_steps = num_val / BATCH_SIZES



    path_prefix = './VOCdevkit/%s/JPEGImages/' % dirName
    gen = Generator(gt, bbox_util, BATCH_SIZES, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    image_path = []
    for train_key in train_keys:
        im = os.path.join(path_prefix, train_key)
        image_path.append(im)

    #logging.info("tainable filenames %s" % (image_path,) )

    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    print( model.summary() )
    model.load_weights('weights_SSD300.hdf5', by_name=True)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
            'conv2_1', 'conv2_2', 'pool2',
            'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
    #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                verbose=1,
                                                save_weights_only=True),
                keras.callbacks.LearningRateScheduler(schedule)]


    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

    nb_epoch = 3
    # keras 1
    #history = model.fit_generator(gen.generate(True), gen.train_batches,
    #                            nb_epoch, verbose=1,
    #                            callbacks=callbacks,
    #                            validation_data=gen.generate(False),
    #                            nb_val_samples=gen.val_batches,
    #                            nb_worker=1)

    # keras 2
    history = model.fit_generator(generator = gen.generate(True), 
                                    steps_per_epoch = steps_per_epoch,
                                epochs=nb_epoch , verbose=1,
                                callbacks=callbacks,
                                validation_data = gen.generate(False),
                                validation_steps = validation_steps )



def main(argv):


    if len(sys.argv) == 1:
        logging.info( 'Usage: # python %s DIRname (eg. VOC2007 or myVOC)' % sys.argv )
        quit()                 

    train_proc(argv[0])



if __name__ == "__main__":
    main(sys.argv[1:])