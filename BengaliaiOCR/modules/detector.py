#-*- coding: utf-8 -*-
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import fastdeploy as fd
import cv2
import copy
import numpy as np
import os
import sys
import tarfile
import requests
from tqdm import tqdm
from .utils import LOG_INFO,create_dir
#-------------------------
# helpers from paddle ppocr network: 
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/network.py
#-------------------------
def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        LOG_INFO("Something went wrong while downloading models",mcolor="red")
        sys.exit(0)


def maybe_download(model_storage_directory, url):
    # using custom model
    tar_file_name_list = ['.pdiparams', '.pdiparams.info', '.pdmodel']
    params_file        = os.path.join(model_storage_directory, 'inference.pdiparams')
    model_file         = os.path.join(model_storage_directory, 'inference.pdmodel')
    if not os.path.exists(params_file) or not os.path.exists(model_file):
        assert url.endswith('.tar'), 'Only supports tar compressed package'
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        print('download {} to {}'.format(url, tmp_path))
        os.makedirs(model_storage_directory, exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, 'r') as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if member.name.endswith(tar_file_name):
                        filename = 'inference' + tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(os.path.join(model_storage_directory, filename),'wb') as f:
                    f.write(file.read())
        os.remove(tmp_path)

#-------------------------
# main class
#-------------------------

class Detector(object):
    def __init__(self,use_gpu=True,
                      device_id=0,
                      max_side_len=960,
                      det_db_thresh=0.3,
                      det_db_box_thresh = 0.6,
                      det_db_unclip_ratio = 1.5,
                      det_db_score_mode = "slow",
                      use_dilation = False,
                      line_model_url='https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                      word_model_url='https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'):
        # set detection options
        self.det_option = fd.RuntimeOption()
        if use_gpu:
            self.det_option.use_gpu(device_id)
        # set processor params
        self.max_side_len           =   max_side_len
        self.det_db_thresh          =   det_db_thresh
        self.det_db_box_thresh      =   det_db_box_thresh
        self.det_db_unclip_ratio    =   det_db_unclip_ratio
        self.det_db_score_mode      =   det_db_score_mode
        self.use_dilation           =   use_dilation
        # model paths
        base_dir = os.path.expanduser("~/.bengali_ai_ocr/")
        line_model_path=create_dir(base_dir,"line")
        word_model_path=create_dir(base_dir,"word")
        maybe_download(line_model_path,line_model_url)
        maybe_download(word_model_path,word_model_url)
        
        # get models
        self.line_model=self.load_model(line_model_path)
        self.word_model=self.load_model(word_model_path)


    def load_model(self,model_path):
        det_model_file = os.path.join(model_path, "inference.pdmodel")
        det_params_file = os.path.join(model_path, "inference.pdiparams")
        det_model = fd.vision.ocr.DBDetector(det_model_file, det_params_file, runtime_option=self.det_option)
        # Set the preporcessing parameters
        det_model.preprocessor.max_side_len         = self.max_side_len
        # Set the postporcessing parameters
        det_model.postprocessor.det_db_thresh       = self.det_db_thresh
        det_model.postprocessor.det_db_box_thresh   = self.det_db_box_thresh
        det_model.postprocessor.det_db_unclip_ratio = self.det_db_unclip_ratio
        det_model.postprocessor.det_db_score_mode   = self.det_db_score_mode
        det_model.postprocessor.use_dilation        = self.use_dilation
        return det_model
    
            
    def get_rotate_crop_image(self,img, points):
        # Use Green's theory to judge clockwise or counterclockwise
        # author: biyanhua
        d = 0.0
        for index in range(-1, 3):
            d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                        points[index + 1][0] - points[index][0])
        if d < 0: # counterclockwise
            tmp = np.array(points)
            points[1], points[3] = tmp[3], tmp[1]

        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height],
                            [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
        
    def get_word_boxes(self,image):
        result = self.word_model.predict(image)
        return result.boxes

    def get_line_boxes(self,image):
        result = self.line_model.predict(image)
        return result.boxes
    
    def get_crops(self,img,boxes):
        '''
            extract locations and crops
        '''
        crops=[]
        for bno in range(len(boxes)):
            tmp_box = copy.deepcopy(boxes[bno])
            x1,y1,x2,y2,x3,y3,x4,y4=tmp_box
            tmp_box=np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],dtype=np.float32)
            img_crop = self.get_rotate_crop_image(img,tmp_box)
            crops.append(img_crop)

        return crops

