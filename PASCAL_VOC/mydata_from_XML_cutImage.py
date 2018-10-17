import numpy as np
import os
from xml.etree import ElementTree

import cv2
import sys
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)



class XML_preprocessor(object):

    def __init__(self, data_path):
        
        
        self.path_prefix = os.path.join( data_path, "Annotations")

        self.new_dir = os.path.join( data_path, "cut_images" )
        self.img_dir = os.path.join( data_path, "JPEGImages" )       


        self.num_classes = 20
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)


        for filename in filenames:

            filename = os.path.join( self.path_prefix , filename )
            tree = ElementTree.parse( filename )
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            


            
            for obj_id, object_tree in enumerate( root.findall('object') ):
                for bounding_box in object_tree.iter('bndbox'):
                    #xmin = float(bounding_box.find('xmin').text)/width
                    #ymin = float(bounding_box.find('ymin').text)/height
                    #xmax = float(bounding_box.find('xmax').text)/width
                    #ymax = float(bounding_box.find('ymax').text)/height

                    xmin = int(bounding_box.find('xmin').text)
                    ymin = int(bounding_box.find('ymin').text)
                    xmax = int(bounding_box.find('xmax').text)
                    ymax = int(bounding_box.find('ymax').text)


                class_name = object_tree.find('name').text
                image_name = root.find('filename').text + ".jpg"
                new_image_name = root.find('filename').text + "_" + str(obj_id) + ".jpg"
                
                if class_name in ["Prescription","Bill"]:

                    imagefile = os.path.join(self.img_dir,image_name)
                    im = cv2.imread(imagefile)
                    
                    if not im is None:
                        im_pre = im[ymin:ymax,xmin:xmax]
                        new_filename = os.path.join(self.new_dir,new_image_name)
                        print("saved.", new_filename)
                        cv2.imwrite(new_filename,im_pre)


                #one_hot_class = self._to_one_hot(class_name)
                #one_hot_classes.append(one_hot_class)
            

            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

## example on how to use it

def main(argv):

    if len(sys.argv) == 1:
        logging.info( 'Usage: # python %s DIRname (eg. VOC2007 or myVOC)' % sys.argv )
        quit()                 

    targetDir = os.path.join(  '../VOCdevkit',  argv[0] )
    data = XML_preprocessor(targetDir).data



if __name__ == "__main__":
    main(sys.argv[1:])


