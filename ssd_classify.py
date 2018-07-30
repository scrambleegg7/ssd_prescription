import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf

import matplotlib.image as mpimg

from ssd import SSD300
from ssd_utils import BBoxUtility
import os
import glob  

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

from PIL import ImageEnhance
from PIL import Image as pil_image

from timeit import default_timer as timer

import logging, sys


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

def makeClassColors():

    class_colors = []
    class_names = ["background", "Prescription", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    NUM_CLASSES = len(class_names)

    for i in range(0, NUM_CLASSES):
        # This can probably be written in a more elegant manner
        hue = 255*i/NUM_CLASSES
        col = np.zeros((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128 # Saturation
        col[0][0][2] = 255 # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)     
    
    return class_colors

def classify():

    class_colors = makeClassColors()

    voc_classes = ['Prescription', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    NUM_CLASSES = len(voc_classes) + 1

    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    #model.load_weights('weights_SSD300.hdf5', by_name=True)
    weights_file = "./checkpoints/weights.00-1.65.hdf5"
    model.load_weights(weights_file, by_name=True)

    bbox_util = BBoxUtility(NUM_CLASSES)

    #target_dir = "/Users/donchan/Documents/myData/miyuki/camera/prescription"
    target_dir = "/Volumes/m1124/FTP/073010"
    
    # load original image
    test_image_file = os.path.join(target_dir,"*.jpg")
    #files = glob.glob("/Volumes/m1124/FTP/073010/*.jpg")

    files = os.listdir(target_dir)
    files = [ os.path.join( target_dir, f  ) for f in files if ".jpg" in f  ]



    logging.info("- "* 40)
    logging.info(test_image_file)
    #logging.info(files)
    logging.info("- "* 40)    
    # build pipeline images for classification (original image size)
    pipeline_images = [ mpimg.imread(file) for file in files ]


    # load image for prediction (shrinked 300 x 300)
    image_load_ops = lambda x:image.load_img(x , target_size=(300, 300))
    image_array_ops = lambda x:image.img_to_array(x)

    inputs = []    
    for x in files:
        img = image.load_img(x , target_size=(300, 300))
        img = image.img_to_array(img)
        inputs.append( img.copy() )
    #inputs = list( map(image_load_ops, files) )
    #inputs = list( map(image_array_ops, inputs) )
    
    # keras module to look in class of data image
    logging.info(" keras model starting..... ")
    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    logging.info("")
    logging.info("Now classification for every images.")
    for i, img in enumerate(pipeline_images):
        # Parse the outputs.
        to_draw = img.copy()
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()
        prescription_label_name = 0

        for j in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[j] * img.shape[1]))
            ymin = int(round(top_ymin[j] * img.shape[0]))
            xmax = int(round(top_xmax[j] * img.shape[1]))
            ymax = int(round(top_ymax[j] * img.shape[0]))
            score = top_conf[j]
            label = int(top_label_indices[j])
            label_name = voc_classes[label - 1]
            if label_name == "Prescription":
                prescription_label_name = 1

            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            logging.info("object NO: %d %s" % ( (j+1), label_name ) )
            logging.info("rectangle info: %s" % (coords,) )
            #logging.info(label_name,color)
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), 
                    class_colors[label], 2)

        if prescription_label_name == 1:
            cv2.imwrite(  os.path.join( "./results", str(i)+'.jpg' )    , to_draw )
        
        #plt.show()    


def main():
    classify()



if __name__ == "__main__":
    main()