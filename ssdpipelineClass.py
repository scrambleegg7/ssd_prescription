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

from ssd_k2 import SSD300
#
# for keras version 1.2
# from ssd import SSD300
#
from ssd_utils import BBoxUtility

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

from PIL import ImageEnhance
from PIL import Image as pil_image

from timeit import default_timer as timer

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))



class SSDPipeline(object):

    def __init__(self):

        voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        NUM_CLASSES = len(voc_classes) + 1

        input_shape=(300, 300, 3)
        self.model = SSD300(input_shape, num_classes=NUM_CLASSES)
        weights_file = "./checkpoints/weights.10-2.85.hdf5"        
        #weights_file = "./checkpoints/weights.39-1.61_ubuntu.hdf5"

        self.model.load_weights(weights_file, by_name=True)
        self.bbox_util = BBoxUtility(NUM_CLASSES)

    def loadImage(self,video_path):

        vid = cv2.VideoCapture(video_path)
        vidw = vid.get(3) # CV_CAP_PROP_FRAME_WIDTH
        vidh = vid.get(4) # CV_CAP_PROP_FRAME_HEIGHT

        print(vidw,vidh)
        input_shape = (300,300,3)
        vidar = vidw/vidh
        #print(vidar)
        return vidar

    def setClassColors(self):

        self.class_colors = []
        self.class_names = ["background", "Prescription", "None", "title", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
        NUM_CLASSES = len(self.class_names)

        for i in range(0, NUM_CLASSES):
            # This can probably be written in a more elegant manner
            hue = 255*i/NUM_CLASSES
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col) 
        
    def pipeline(self,orig_image):
        
        start_frame = 0

        # this is manual adjustment parameter
        # For binary classifilcation, set higher threshhold rather than 0.5
        conf_thresh = 0.50

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        vidh, vidw, _ = orig_image.shape
        vidar = vidw/vidh

        input_shape = (300,300,3)
        display_shape = (600,600,3)
        
        im_size = (input_shape[0], input_shape[1])   
        resized = cv2.resize(orig_image, im_size)
        to_draw = cv2.resize(resized, (int(input_shape[0]*vidar), input_shape[1]))
        #to_draw = cv2.resize(resized, (int(display_shape[0]*vidar), display_shape[1]))

        #to_draw = orig_image.copy()

        # Use model to predict 
        inputs = [image.img_to_array(resized)]
        tmp_inp = np.array(inputs)
        x = preprocess_input(tmp_inp)
        y = self.model.predict(x)
        
        #preds = model.predict(inputs, batch_size=1, verbose=1)
        results = self.bbox_util.detection_out(y)
        
        if len(results) > 0 and len(results[0]) > 0:
            # Interpret output, only one frame is used 
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin = results[0][:, 2]
            det_ymin = results[0][:, 3]
            det_xmax = results[0][:, 4]
            det_ymax = results[0][:, 5]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]


            classes = []
            probs = []
            for i in range(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                # Draw the box on top of the to_draw image
                class_num = int(top_label_indices[i])
                
                #  sorry, but x length bigger than half of screen size avoid to 
                #  draw rectangle 
                if ( abs(xmax-xmin) > to_draw.shape[1] / 2. ):
                    continue 

                classes.append(self.class_names[class_num])
                probs.append(top_conf[i])

                cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), 
                            self.class_colors[class_num], 2)
                text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])

                text_top = (xmin, ymin-10)
                text_bot = (xmin + 80, ymin + 5)
                text_pos = (xmin + 5, ymin)
                cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)
                cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                
        # Calculate FPS
        # This computes FPS for everything, not just the model's execution 
        # which may or may not be what you want
        #curr_time = timer()
        #exec_time = curr_time - prev_time
        #prev_time = curr_time
        #accum_time = accum_time + exec_time
        #curr_fps = curr_fps + 1
        #if accum_time > 1:
        #    accum_time = accum_time - 1
        #    fps = "FPS: " + str(curr_fps)
        #    curr_fps = 0

        # Draw FPS in top left corner
        #cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
        #cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

        #print("object NO:", i+1)
        #print("rectangle info: ", coords)
        
        
        return to_draw, classes, probs
