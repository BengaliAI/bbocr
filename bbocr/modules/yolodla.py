#-*- coding: utf-8 -*-
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import os
import math
from pathlib import Path
from ultralytics import YOLO
# from .utils import download
# from .modules import LayoutAnalyzer
#--------------debug-------------------
import gdown
from abc import ABCMeta, abstractmethod
def download(id,save_dir):
    gdown.download(id=id,output=save_dir,quiet=False)
class LayoutAnalyzer(metaclass=ABCMeta):
    """layout analyzer base class
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def get_rois(self):
        pass
#--------------debug-------------------

class YoloDLA(LayoutAnalyzer):
    def __init__(self,
                 yolo_dla_gid="1n-XbOwUwgMjaFPFzEJ59Avrl9Nc5xsx8",
                 names = {0: 'paragraph', 
                          1: 'text_box', 
                          2: 'image', 
                          3: 'table'}):
        super().__init__()
        self.yolo_dla_gid=yolo_dla_gid
        self.weight_path=self.get_model_weights()
        self.model=YOLO(self.weight_path)
        self.names=names
    #-----------------------------------------------------
    def get_model_weights(self):
        home_path = str(Path.home())
        weight_path = Path(
            home_path,
            ".bengali_ai_ocr",
            "best.pt"
        )
        weight_path = Path(weight_path).resolve()
        weight_path.parent.mkdir(exist_ok=True, parents=True)
        weight_path = str(weight_path)
        if not os.path.isfile(weight_path):
            download(self.yolo_dla_gid,weight_path)
        return weight_path
    
    #-----------------------------------------------------
    def get_yolo_prediction(self,image):
        return self.model(image)[0]
    #-----------------------------------------------------
    def get_normalized_coordinates(self,xyxy_tensor, height, width):
        x_min = xyxy_tensor[0][0].item() / width
        y_min = xyxy_tensor[0][1].item() / height
        x_max = xyxy_tensor[0][2].item() / width
        y_max = xyxy_tensor[0][3].item() / height
        coordinates = [x_min, y_min, x_max, y_max]
        return coordinates
    
    def get_original_coordinates(self,normalized_coordinates, image_width, image_height):
        orig_coordinates = [None]*4
        orig_coordinates[0] = math.floor(normalized_coordinates[0] * image_width)
        orig_coordinates[1] = math.floor(normalized_coordinates[1] * image_height)
        orig_coordinates[2] = math.ceil(normalized_coordinates[2] * image_width)
        orig_coordinates[3] = math.ceil(normalized_coordinates[3] * image_height)
        return orig_coordinates
    #-----------------------------------------------------
    def get_rois(self,image):
        # get results class of yolo
        res = self.get_yolo_prediction(image)
        # data containers
        region_of_interests = []
        cropped_text_region_all_boxes = []
        # run for all boxes
        for i in range(len(res.boxes)):
            # dictionary to store information
            info_dict = {"class": None,
                        "coordinates": None,
                        "left": None,
                        "top": None,
                        "elem_height": None,
                        "elem_width": None,
                        "img_height": None,
                        "img_width": None,
                        "text": '',
                        "single-line": False,
                        "img_src": None}
            # get image height width of the bbox
            img_height, img_width = res.boxes[i].orig_shape
            # get actual co-ordinates
            normalized_coordinates = self.get_normalized_coordinates(res.boxes[i].xyxy, img_height, img_width)
            # get the class
            info_dict['class'] = self.names[res.boxes[i].cls.item()]
            # save localization values
            info_dict['coordinates'] = normalized_coordinates
            info_dict['left'], info_dict['top'] = normalized_coordinates[0]*100, normalized_coordinates[1]*100
            info_dict['img_height'], info_dict['img_width'] = img_height, img_width
            info_dict['elem_width'] = (normalized_coordinates[2] - normalized_coordinates[0]) * 100
            info_dict['elem_height'] = (normalized_coordinates[3] - normalized_coordinates[1]) * 100


#-----------------debug---------------------------------
if __name__=="__main__":
    import cv2
    image=cv2.imread("/home/ansary/WORK/Bengaliai/ocr/tests/dla.png") 
    dla=YoloDLA()
    dla.get_rois(image)