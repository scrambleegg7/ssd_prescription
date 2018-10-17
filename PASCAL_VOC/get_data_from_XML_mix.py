import numpy as np
import os
from xml.etree import ElementTree
import pandas as pd   

import logging, sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 20
        self.data = dict()

        # open object checker file

        boat_file = '../VOCdevkit/VOC2007/ImageSets/Main/boat_trainval.txt'
        bus_file = '../VOCdevkit/VOC2007/ImageSets/Main/bus_trainval.txt'
        
        self.df_boat = pd.read_csv(boat_file, header=None, sep="\s+", names=["filename", "flag"], dtype={'filename':'object'} )
        self.df = self.df_boat[self.df_boat.flag == 1].copy()
        self.df_bus = pd.read_csv(bus_file, header=None, sep="\s+", names=["filename", "flag"], dtype={'filename':'object'} )
        self.df_bus = self.df_bus[self.df_bus.flag == 1].copy()
        
        self.df = self.df.append( self.df_bus )
        
        logging.info( self.df.shape)

        self._preprocess_XML()


    def checkBoat(self, filename):

        filename = filename.split('.')[0]
        #print(filename)
        
        if filename in self.df["filename"].tolist():
            return True
        else:
            return False


    def _preprocess_XML(self):


        filenames = os.listdir(self.path_prefix)
        for filename in filenames:

            #if not self.checkBoat(filename):
            #    if not "image18" in filename:
            #        continue
            #logging.info("boat file: %s" %    filename)

            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            if "image18" in filename:
                image_name += ".jpg"
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name == 'Prescription':
            one_hot_vector[0] = 1
        elif name == 'None':
            one_hot_vector[1] = 1
        elif name == 'title':
            one_hot_vector[2] = 1
        elif name == 'Documents':
            one_hot_vector[3] = 1
        elif name == 'item':
            one_hot_vector[4] = 1
        elif name == 'bus':
            one_hot_vector[5] = 1
        elif name == 'car':
            one_hot_vector[6] = 1
        elif name == 'cat':
            one_hot_vector[7] = 1
        elif name == 'chair':
            one_hot_vector[8] = 1
        elif name == 'cow':
            one_hot_vector[9] = 1
        elif name == 'diningtable':
            one_hot_vector[10] = 1
        elif name == 'dog':
            one_hot_vector[11] = 1
        elif name == 'horse':
            one_hot_vector[12] = 1
        elif name == 'motorbike':
            one_hot_vector[13] = 1
        elif name == 'person':
            one_hot_vector[14] = 1
        elif name == 'pottedplant':
            one_hot_vector[15] = 1
        elif name == 'sheep':
            one_hot_vector[16] = 1
        elif name == 'sofa':
            one_hot_vector[17] = 1
        elif name == 'train':
            one_hot_vector[18] = 1
        elif name == 'tvmonitor':
            one_hot_vector[19] = 1
        elif name == 'aeroplane':
            pass
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

## example on how to use it
import pickle
data = XML_preprocessor('../VOCdevkit/Prescription/Annotations/').data
#data = XML_preprocessor('../VOCdevkit/VOC2007/Annotations/').data
pickle.dump(data,open('prescription.pkl','wb'))


