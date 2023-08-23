# %%
# !nvcc --version
# import torch, torchvision,numpy
# print(torch.__version__, torch.cuda.is_available(), numpy.__version__) #1.13.1 True 1.23.5

# %%
import layoutparser as lp
import cv2
import yaml
import io
import os
from ultralytics import YOLO
from pathlib import Path

# %%
# from paddleocr import PaddleOCR
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
from pprint import pprint
import os
import time


import fastdeploy as fd
import cv2
import os
from multiprocessing import Pool

import pickle

# LOG_ALL = True
LOG_ALL = False


roi_directory = r'data/roi'
paragraph_wise_predicted_word_box_directory = r'data/paragraph_wise_predicted_word_box'
predicted_text_directory = r'data/predicted_text'
word_annotation_directory = r'data/word_annotation'



# roi_suffix = '_region_of_interests.pkl'
# pred_text_suffix = '_paragraph_wise_recognized_text.pkl'
# paragraph_wise_predicted_word_box_suffix = '_paragraph_wise_word_boxes.pkl'
# word_gc_all_box_all_lines_suffix = '_word_gc_all_box_all_lines.pkl'

pred_text_suffix = f'{preproc_suffix}_paragraph_wise_recognized_text.pkl'
paragraph_wise_predicted_word_box_suffix = f'_{preproc_suffix}_paragraph_wise_word_boxes.pkl'
roi_suffix = f'{preproc_suffix}_region_of_interests.pkl'
word_gc_all_box_all_lines_suffix = f'_{preproc_suffix}_word_gc_all_box_all_lines.pkl'

badlad_coco_path = r'data/badlad-test-coco.json'
word_gc_all_box_all_lines_directory = r'data/word_gc_all_box_all_lines'

# pickle_dump(standalone_all_pred_word_bboxes, )
    # pickle_dump(standalone_all_pred_word_texts, )

standalone_all_pred_word_bboxes_dir = 'data/standalone_all_pred_word_bboxes'
standalone_all_pred_word_bboxes_suffix = '_standalone_all_pred_word_bboxes.pkl'

standalone_all_pred_word_texts_dir = 'data/standalone_all_pred_word_texts'
standalone_all_pred_word_texts_suffix = '_standalone_all_pred_word_texts.pkl'





def pickle_dump(object, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(object, file)

def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=False, help="Path of Detection model of PPOCR.",
        default=f'/home/{uname}/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer')
    parser.add_argument(
        "--det_model2", required=False, help="Path of Detection model of PPOCR.",
        default=f'/home/{uname}/.paddleocr/whl/det/ml/Multilingual_PP-OCRv3_det_infer')
    
    parser.add_argument(
        "--rec_model", required=False, help="Path of Detection model of PPOCR.",
        default=f'/home/{uname}/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_infer')
    # parser.add_argument(
        # "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='gpu',
        # default='cpu',
        help="Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    return parser.parse_args()


def build_option(args):

    det_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        det_option.use_gpu(args.device_id)

    return det_option


args = parse_arguments()

def load_yolo():
    global yolomodel
    yolomodel = YOLO(model_weight)
def load_bocr():
    global rec
    rec = BanglaOCR(ONNX_PATH)

def load_model():
  det_model_file = os.path.join(args.det_model, "inference.pdmodel")
  det_params_file = os.path.join(args.det_model, "inference.pdiparams")
  
#   rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
#   rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
#   rec_label_file = 'labels.txt'
  
#   rec_option = build_option(args)
  
#   global rec_model
  # rec_model = fd.vision.ocr.Recognizer(
    # rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)
  # Set the runtime option
  det_option = build_option(args)

  # Create the det_model
  global det_model
  det_model = fd.vision.ocr.DBDetector(
      det_model_file, det_params_file, runtime_option=det_option)

  # Set the preporcessing parameters
  det_model.preprocessor.max_side_len = 960
  det_model.postprocessor.det_db_thresh = 0.3
  det_model.postprocessor.det_db_box_thresh = 0.6
  det_model.postprocessor.det_db_unclip_ratio = 1.5
  det_model.postprocessor.det_db_score_mode = "slow"
  det_model.postprocessor.use_dilation = False


def load_model2():
  det_model_file = os.path.join(args.det_model2, "inference.pdmodel")
  det_params_file = os.path.join(args.det_model2, "inference.pdiparams")
  
#   rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
#   rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
#   rec_label_file = 'labels.txt'
  
#   rec_option = build_option(args)
  
#   global rec_model
  # rec_model = fd.vision.ocr.Recognizer(
    # rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)
  # Set the runtime option
  det_option = build_option(args)

  # Create the det_model
  global det_model2
  det_model2 = fd.vision.ocr.DBDetector(
      det_model_file, det_params_file, runtime_option=det_option)

  # Set the preporcessing parameters
  det_model2.preprocessor.max_side_len = 960
  det_model2.postprocessor.det_db_thresh = 0.3
  det_model2.postprocessor.det_db_box_thresh = 0.6
  det_model2.postprocessor.det_db_unclip_ratio = 1.5
  det_model2.postprocessor.det_db_score_mode = "slow"
  det_model2.postprocessor.use_dilation = False
# %%
# line=PaddleOCR(use_angle_cls=False, lang='en',use_gpu=True, use_dilation=True) #, det_model_dir=r'C:\\Users\\Admin/.paddleocr/whl\\det\\en\\en_PP-OCRv3_det_slim_infer')
# word=PaddleOCR(use_angle_cls=False, lang='ar',use_gpu=True, use_dilation=True) #, det_model_dir=r'C:\\Users\\Admin/.paddleocr/whl\\det\\ml\\Multilingual_PP-OCRv3_det_slim_infer')

# %%
# from __future__ import print_function
#-------------------------
# imports
#-------------------------
import onnxruntime as ort
import numpy as np
import cv2
from bnunicodenormalizer import Normalizer
NORM=Normalizer()

# TensorrtExecutionProvider

# %%
class BanglaOCR(object):
    def __init__(self,
                model_weights,
                providers=['CUDAExecutionProvider'],
                # providers=['CPUExecutionProvider'],
                # providers=['TensorrtExecutionProvider'],
                img_height=32,
                img_width=256,
                pos_max=40):
        self.img_height=img_height
        self.img_width =img_width
        self.pos_max   =pos_max
        self.model     =ort.InferenceSession(model_weights, providers=providers)
        self.vocab     =["blank","!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","।",
                        "ঁ","ং","ঃ","অ","আ","ই","ঈ","উ","ঊ","ঋ","এ","ঐ","ও","ঔ",
                        "ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ","ট","ঠ","ড","ঢ","ণ","ত","থ","দ","ধ","ন",
                        "প","ফ","ব","ভ","ম","য","র","ল","শ","ষ","স","হ",
                        "া","ি","ী","ু","ূ","ৃ","ে","ৈ","ো","ৌ","্",
                        "ৎ","ড়","ঢ়","য়","০","১","২","৩","৪","৫","৬","৭","৮","৯","‍","sep","pad"]

    def process_batch(self,crops):
        batch_img=[]
        batch_pos=[]
        for img in crops:
            # correct padding
            img,_=correctPadding(img,(self.img_height,self.img_width))
            # normalize
            img=img/255.0
            # extend batch
            img=np.expand_dims(img,axis=0)
            batch_img.append(img)
            # pos
            pos=np.array([[i for i in range(self.pos_max)]])
            batch_pos.append(pos)
        # stack
        img=np.vstack(batch_img)
        img=img.astype(np.float32)
        pos=np.vstack(batch_pos)
        pos=pos.astype(np.float32)
        # batch inp
        return {"image":img,"pos":pos}

    # def __call__(self,crops,batch_size=32):
    # def __call__(self,crops,batch_size=8):
    # def __call__(self,crops,batch_size=128): # TODO: final
    def __call__(self,crops,batch_size=128+32): # TODO: final, fist 1 sec benchmark
    # def __call__(self,crops,batch_size=128+64): # TODO: final
    # def __call__(self,crops,batch_size=256): # TODO: final
        # adjust batch_size
        if len(crops)<batch_size:
            batch_size=len(crops)
        if batch_size == 0:
            return []
        texts=[]
        for idx in range(0,len(crops),batch_size):
            batch=crops[idx:idx+batch_size]
            inp=self.process_batch(batch)
            preds=self.model.run(None,inp)[0]
            preds =np.argmax(preds,axis=-1)
            # decoding
            for pred in preds:
                label=""
                for c in pred[1:]:
                    if c!=self.vocab.index("sep"):
                        label+=self.vocab[c]
                    else:
                        break
                texts.append(label)
        texts=[NORM(text)["normalized"] for text in texts]
        texts=[text if text is not None else '' for text in texts]
        return texts

# %%
# from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import cv2
import numpy as np
import copy

# %%
def padWordImage(img,pad_loc,pad_dim,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_val :       the value to pad
    '''

    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        # pad widths
        pad_width =pad_dim-w
        # pads
        pad =np.ones((h,pad_width,3))*pad_val
        # pad
        img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")

# %%
def correctPadding(img,dim,pvalue=255):
    '''
        corrects an image padding
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape

    w_new=int(img_height* w/h)
    img=cv2.resize(img,(w_new,img_height))
    h,w,d=img.shape

    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w)
        img=cv2.resize(img,(img_width,h_new))
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_val=pvalue)
        mask=w

    # error avoid
    img=cv2.resize(img,(img_width,img_height))

    return img,mask

# %%
class Detector(object):
    def __init__(self):
        '''
            initializes a dbnet detector model
        '''
        self.call_rec="paddle"

    def sorted_boxes(self,dt_boxes,dist=10):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < dist and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

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


    def __call__(self,img,result):
        '''
            extract locations and crops
        '''
        boxes= np.array(result, dtype=np.float32)

        # boxes=self.sorted_boxes(boxes) # This existed in the original code

        crops=[]
        for bno in range(len(boxes)):
            tmp_box = copy.deepcopy(boxes[bno])
            img_crop = self.get_rotate_crop_image(img,tmp_box)
            crops.append(img_crop)
        #mask=create_mask(img,boxes)
        #return mask,boxes,crops
        return boxes,crops

# %% [markdown]
# Word Recognition

# %%
def line_segmentation(image):
    result_line = line.ocr(image,rec=False,cls=False)
    return result_line

def word_segmentation(image):
    result_word = word.ocr(image,rec=False,cls=False)
    return result_word

# %%
# def quantize_onnx_model(onnx_model_path, quantized_model_path):
#     from onnxruntime.quantization import quantize_dynamic, QuantType
#     import onnx
#     onnx_opt_model = onnx.load(onnx_model_path)
#     quantize_dynamic(onnx_model_path,
#                      quantized_model_path,
#                      weight_type=QuantType.QUInt8)

#     print(f"quantized model saved to:{quantized_model_path}")

# quantize_onnx_model("bnocr.onnx", "quantized_bnocr.onnx")

# %%


# %%
global det
global rec

det=Detector()
# rec=BanglaOCR(ONNX_PATH)


def word_horizontal_dilation(boxes, image, vertical_dilation = False, ratio = 1.1):
    crops = []
    length = len(boxes)
    height, width, channels = image.shape
    # import pdb; pdb.set_trace()

    for i in range(len(boxes)):

        if i+1 < length:
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] = boxes[i]
            [[x_min1, y_min1], [x_max1, y_min1], [x_max1, y_max1], [x_min1, y_max1]] = boxes[i+1]

            right_gap = x_min1 - x_max
            if right_gap > 0:
                x_max += right_gap // 2
                x_min1 -= right_gap //2

                x_max = min(x_max, width)
                x_min1 = max(x_min1, 0)

                boxes[i][1][0], boxes[i][2][0] = x_max, x_max
                boxes[i+1][0][0], boxes[i+1][3][0] = x_min1, x_min1
            
            if vertical_dilation:
                top_padding = int((y_max - y_min)*ratio)
                y_min_padded = y_min-top_padding
                y_max_padded = y_max+top_padding

                y_max_padded = min(y_max_padded, height)
                y_min_padded = max(y_min_padded, 0)

                boxes[i][0][1], boxes[i][1][1] = (y_min_padded), (y_min_padded)
                boxes[i][2][1], boxes[i][3][1] = (y_max_padded), (y_max_padded)

                crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                h,w,d = crop.shape
                if h!=0 and w!=0:
                    crops.append(crop)

    crop = image[int(boxes[-1][0][1]):int(boxes[-1][2][1]), int(boxes[-1][0][0]):int(boxes[-1][1][0])]
    h, w, d = crop.shape
    if h!=0 and w!=0:
        crops.append(crop)
    # print('shape of crops', len(crops), length)
    return crops, boxes


def crop_word_regions(image, result_word):
    boxes,crops=det(image, result_word[0])
    return crops, boxes


def recognize_word(crops):
    texts = rec(crops)
    return texts

def dla_predict(image):
    global yolomodel
    res = yolomodel(image, verbose=False)[0]
    return res

# %% [markdown]
# YOLOv8 Inference

# %% [markdown]
# # Visualize Object Detection

# %%
def yolo(model_weight,image_path):
    # model = YOLO(model_weight)
    image = cv2.imread(image_path)

    # upscaling 2x
    # height, width = image.shape[:2]
    # image = cv2.resize(image, (2*width, 2*height))


    # plt.imshow(image)
    color_map = {
        'text_box':   'red',
        'paragraph':  'blue',
        'image':   'green',
        'table':  'yellow',
    }

    # layout_predicted = model(image)
    # res = model(image)
    res = pool0.map(dla_predict, [image])[0]
    
    res_plotted = res[0].plot(conf=False, labels=False)

    resized = cv2.resize(res_plotted, (500, 500))

    cv2.imshow('Resized Image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res


def crop_all_text_box(image_path, res):
    croped_imgs=[]
    image = cv2.imread(image_path)
    for i in range(len(res[0].boxes)):

        x = int(res[0].boxes[i].xyxy[0][0])
        y = int(res[0].boxes[i].xyxy[0][1])
        width = int(res[0].boxes[i].xyxy[0][2] - res[0].boxes[i].xyxy[0][0])
        height = int(res[0].boxes[i].xyxy[0][3] - res[0].boxes[i].xyxy[0][1])

        crop_img = image[y:y+height, x:x+width]
        croped_imgs.append(crop_img)
        cv2.imshow('Document Layout Detected', crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return croped_imgs


def single_image_layout(model_weight,image_path,config_yml):
    if len(config_yml)==0:
        return yolo(model_weight,image_path)
    # else:
    #     return run_rcnn_model(model_weight,image_path,config_yml)


# %%
# image_path = 'test_images\\two_col.png'
config_yml = ''

# model_weight = '/mnt/hdd/jawaril/src/distillation/dla/weights/yolo/best.pt'


# result = single_image_layout(model_weight,image_path,config_yml)

# %% [markdown]
# # HTML Mapping

# %%
from bs4 import BeautifulSoup
from bs4.formatter import HTMLFormatter

class HtmlGenerator:
    def __init__(self, filename="default"):
        with open("reconstruction/templates/index.html", 'r') as f:
            index_template = f.read()

        self.index_template = BeautifulSoup(index_template, "html.parser")
        self.index_template_root_div = self.index_template.find("div", {"id": "root"})
        self.filename = filename

    def read_html_template(self, template_name):
        with open(f"reconstruction/templates/{template_name}.html", 'r') as f:
            template = f.read()
            soup_template = BeautifulSoup(template, "html.parser")
            return soup_template


    def get_styles(self, dict):
        styles = f'top: {dict["top"]}vh; left: {dict["left"]}vw; height: {dict["elem_height"]}vh; width: {dict["elem_width"]}vw;'
        return styles


    def insert_paragraph(self, paragraph_info):
        paragraph_template = self.read_html_template("paragraph")

        p_tag = paragraph_template.find('p')
        text = paragraph_template.new_string(paragraph_info['text'])
        p_tag.append(text)

        paragraph_div = paragraph_template.find("div")
        paragraph_div["style"] = self.get_styles(paragraph_info)

        self.index_template_root_div.append(paragraph_template)


    def insert_text_box(self, text_box_info):
      text_box_template = self.read_html_template("text_box")

      p_tag = text_box_template.find('p')
      text = text_box_template.new_string(text_box_info['text'])
      p_tag.append(text)

      text_box_div = text_box_template.find("div")
      text_box_div["style"] =self.get_styles(text_box_info)

      self.index_template_root_div.append(text_box_template)


    def insert_image(self, img_info):

        image_template = self.read_html_template("image")

        img_div = image_template.find("div")
        img_div["style"] = self.get_styles(img_info)

        img_tag = image_template.new_tag('img')
        img_tag['src'] = img_info['img_src']

        img_style = "width: 100%; height: 100%; object-fit: fill;"
        img_tag['style'] = img_style

        img_div.append(img_tag)

        self.index_template_root_div.append(image_template)


    def create_html_file(self):
        global img_src_save_dir
        html_path = Path(img_src_save_dir).parent
        with open(html_path/f"{self.filename}.html", "w") as f:
            f.write(str(self.index_template.prettify(formatter=HTMLFormatter(indent=2))))

# %%
def generate_html(detected_elements_info, file_name):
    file_name, extension = file_name.split(".")

    gen = HtmlGenerator(file_name)

    for element_info in detected_elements_info:

        if element_info['class'] == 'paragraph':
            gen.insert_paragraph(element_info)

        elif element_info['class'] == 'text_box':
            gen.insert_text_box(element_info)

        elif element_info['class'] == 'image':
            gen.insert_image(element_info)

    gen.create_html_file()

# %%
import math
from shapely.geometry import box
import time

config_yml = ''
# model_weight ='best.pt'
# model_weight = 


def get_normalized_coordinates(xyxy_tensor, height, width):
    x_min = xyxy_tensor[0][0].item() / width
    y_min = xyxy_tensor[0][1].item() / height
    x_max = xyxy_tensor[0][2].item() / width
    y_max = xyxy_tensor[0][3].item() / height

    coordinates = [x_min, y_min, x_max, y_max]
    return coordinates


def get_original_coordinates(normalized_coordinates, image_width, image_height):
    orig_coordinates = [None]*4

    orig_coordinates[0] = math.floor(normalized_coordinates[0] * image_width)
    orig_coordinates[1] = math.floor(normalized_coordinates[1] * image_height)
    orig_coordinates[2] = math.ceil(normalized_coordinates[2] * image_width)
    orig_coordinates[3] = math.ceil(normalized_coordinates[3] * image_height)

    return orig_coordinates


def get_coordinates_from_segmentation(result_word):
    words_xyxy = []

    for i in range(len(result_word[0])):
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] = result_word[0][i]
        words_xyxy.append([math.floor(x_min), math.floor(y_min), math.ceil(x_max), math.ceil(y_max)])

    return words_xyxy

# line_coordinates, line_coordinates_gc = get_coordinates_from_segmentation_fd(result_line, padding_amount_box)

def get_coordinates_from_segmentation_fd(result_word, padding_amount_box, global_coordinates_box):
    words_xyxy = []
    words_xyxy_gc = []

    for i in range(len(result_word)):
        [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max] = result_word[i]
        word_xyxy = [math.floor(x_min), math.floor(y_min), math.ceil(x_max), math.ceil(y_max)]
        word_xyxy_gc = [
            global_coordinates_box[0] + word_xyxy[0], 
            global_coordinates_box[1] + word_xyxy[1] - padding_amount_box[0], 
            global_coordinates_box[0] + word_xyxy[2], 
            global_coordinates_box[1] + word_xyxy[3] - padding_amount_box[1]
        ]

        words_xyxy.append(word_xyxy)
        words_xyxy_gc.append(word_xyxy_gc)

    return words_xyxy, words_xyxy_gc


def line_vertical_dilation(line_coordinates, img_height, rate=0.1):
    # TODO: i == 0 and i == len() is buggy, use low rates
    # dilation_all_lines = [[0, 0]]*len(line_coordinates)
    dilation_all_lines = [[0, 0] for _ in range(len(line_coordinates))]
    # import pdb; pdb.set_trace()
    # print("line_coordinates", line_coordinates)
    for i in range(len(line_coordinates)):
        if i==0:
            [x_min, y_min, x_max, y_max] = line_coordinates[i]
            # top_padding = int((y_max - y_min)/10)
            top_padding = int((y_max - y_min)*rate)
            # line_coordinates[i][1] = max(y_min - top_padding, 0)
            
            if y_min - top_padding > 0:
                line_coordinates[i][1] = y_min - top_padding
                dilation_all_lines[i][0] = top_padding
            else:
                line_coordinates[i][1] = 0
                dilation_all_lines[i][0] = y_min - 0

            # import pdb; pdb.set_trace()

        elif i==len(line_coordinates)-1:
            [x_min, y_min, x_max, y_max] = line_coordinates[i]
            # bottom_padding = int((y_max - y_min)/10)
            bottom_padding = int((y_max - y_min)*rate)
            
            # line_coordinates[i][3] = min(y_max + bottom_padding, img_height)
            if y_max + bottom_padding < img_height:
                line_coordinates[i][3] = y_max + bottom_padding
                dilation_all_lines[i][1] = bottom_padding
                # dil_bot = bottom_padding
            else:
                line_coordinates[i][3] = img_height
                dilation_all_lines[i][1] =  img_height - y_max
                # dil_bot = 

        if i+1 < len(line_coordinates):
        # if 0:
            [x_min, y_min, x_max, y_max] = line_coordinates[i]
            [x_min1, y_min1, x_max1, y_max1] = line_coordinates[i+1]

            bottom_gap = y_min1 - y_max
            if bottom_gap > 0:
                y_max += bottom_gap // 2
                dilation_all_lines[i][1] = bottom_gap // 2
                y_min1 -= bottom_gap //2
                dilation_all_lines[i+1][0] = bottom_gap // 2
                
                # line_coordinates[i][3] = min(y_max, img_height)
                line_coordinates[i][3] = y_max
                # line_coordinates[i+1][1] = max(y_min1, 0)
                line_coordinates[i+1][1]  = y_min1

    # print("line_coordinates, dilation_all_lines", line_coordinates, dilation_all_lines); exit()
    return line_coordinates, dilation_all_lines


def line_horizontal_dilation(line_coordinates, img_width, rate=0.3):
    # dilation_all_lines = [[0, 0]]*len(line_coordinates)
    horizontal_dilation_all_lines = [[0, 0] for _ in range(len(line_coordinates))]
    # import pdb; pdb.set_trace()
    for i in range(len(line_coordinates)):
        [x_min, y_min, x_max, y_max] = line_coordinates[i]
        # padding = int((y_max - y_min)*0.1)
        padding = int((y_max - y_min)*rate)

        if x_min - padding > 0:
            line_coordinates[i][0] = x_min - padding
            horizontal_dilation_all_lines[i][0] = padding
        else:
            line_coordinates[i][0] = 0
            horizontal_dilation_all_lines[i][0] = x_min
        

        if x_max + padding < img_width:
            line_coordinates[i][2] = x_max + padding
            horizontal_dilation_all_lines[i][1] = padding
            # dil_bot = bottom_padding
        else:
            line_coordinates[i][2] = img_width
            horizontal_dilation_all_lines[i][1] =  img_width - x_max


    # print("line_coordinates, dilation_all_lines", line_coordinates, dilation_all_lines); exit()
    return line_coordinates, horizontal_dilation_all_lines


def top_bottom_padding(
        cropped_text_region, 
        # image_shape, global_coordinates, 
        ratio=1.2):
    # orig_img_height, orig_img_width, _ = image_shape
    h, w = cropped_text_region.shape[:2]
    padded_height = int(h * ratio)
    padded_width = w

    padded_image = np.ones((padded_height, padded_width, 3), dtype=np.uint8) * 255
    # #####
    # -----------------------
    # |  
    # |  --------------
    # |  |

    top_padding = (padded_height - h) // 2
    bottom_padding = top_padding + h

    padded_image[top_padding:bottom_padding, :] = cropped_text_region

    return padded_image, (top_padding, top_padding)


def merge_image_arrays(element_wise_crop_list):
    merged_array = []
    for word_crops in element_wise_crop_list:
        for crop in word_crops:
            merged_array.append(crop)

    return merged_array


def four_tuple_to_xyxy_box(four_tuple_box):
    '''
    [[203.,   8.], [255.,   8.], [255.,  25.], [203.,  25.]] => [x, y, h, w]

    '''
    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] = four_tuple_box

    return list(map(int, [x_min, y_min, x_max, y_max]))


# %%
names = {0: 'paragraph', 1: 'text_box', 2: 'image', 3: 'table'}




import matplotlib.pyplot as plt
import matplotlib.patches as patches

def viz_all_dlabox(image, all_gt_text_region_bbox, prefix):
    for idx, dla_box in enumerate(all_gt_text_region_bbox):
        x, y, width, height = dla_box
        width = width - x
        height = height - y
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image)
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.savefig(f'Evaluation_Metric/tmp/{prefix}_{idx}.png', pad_inches=0.2)
        plt.close()

def viz_wordboxes_in_all_dlabox(image, all_gt_text_region_bbox, wordboxes_in_all_dlabox, prefix="", is_input_xyxy=True):
    for idx, (dla_box, word_boxes) in enumerate(zip(all_gt_text_region_bbox, wordboxes_in_all_dlabox)):
        if is_input_xyxy:
            x, y, width, height = dla_box
            width = width - x
            height = height - y
        else:
            x, y, width, height = dla_box
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image)
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for word_box in word_boxes:
            # import pdb; pdb.set_trace()
            x, y, width, height = word_box
            width = width - x
            height = height - y
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            # break
        
        plt.savefig(f'Evaluation_Metric/tmp/{prefix}_{idx}.png', pad_inches=0.2)
        plt.close()

def viz_wordboxes_in_all_dlabox_all_words(image, all_gt_text_region_bbox, wordboxes_in_all_dlabox, word_gc_all_box_all_lines, prefix=""):
    # import pdb; pdb.set_trace()
    for idx, (dla_box, word_boxes, gc_box_all_lw) in enumerate(zip(all_gt_text_region_bbox, wordboxes_in_all_dlabox, word_gc_all_box_all_lines)):
        x, y, width, height = dla_box
        width = width - x
        height = height - y
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image)
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for word_box, gc_box_all_w in zip(word_boxes, gc_box_all_lw):
            x, y, width, height = word_box
            width = width - x
            height = height - y
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            # break

            for gc_box_this_w in gc_box_all_w:
            # if 1:
                if len(gc_box_this_w) == 0: continue
                x, y, width, height = gc_box_this_w
                width = width - x
                height = height - y
                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

        
        plt.savefig(f'Evaluation_Metric/tmp/{prefix}_{idx}.png', pad_inches=0.2)
        plt.close()

@profile
def run_yolo_model(model_weight, image_path, file_name, img_src_save_directory, standalone_OCR_required=False):
    file_name, extension = file_name.split('.')
    # model = YOLO(model_weight)
    image = cv2.imread(image_path)
    # upscaling 2x
    # height, width = image.shape[:2]
    # image = cv2.resize(image, (2*width, 2*height))
    # plt.imshow(image)
    color_map = {
        'text_box':   'red',
        'paragraph':  'blue',
        'image':   'green',
        'table':  'yellow',
    }
    # layout_predicted = model(image)
    # res = model(image)[0]
    runtime = {}
    start = time.time()
    res = pool0.map(dla_predict, [image])[0]
    dur = time.time() - start
    runtime['dla'] = dur
    # Visualize dla inference
    res_plotted = res.plot(conf=False, labels=False)
    resized = cv2.resize(res_plotted, (500, 500))
    cv2.imwrite(f'dla_inference/{file_name}.jpg', resized)
    
    # template
    # start = time.time()
    # dur = time.time() - start
    # runtime['___'] = dur

    start = time.time()
    region_of_interests = []
    num_of_words = []
    word_crops = []
    word_boxes = []
    all_result_lines = []
    counter = 0
    line_counter = 0
    cropped_text_region_all_boxes = []
    padding_amount_all_boxes = []
    global_coordinates_all_box = []
    info_dict_all_boxes = []


    

    for i in range(len(res.boxes)):
        line_counter = 0
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
                    "img_src": None
                    }

        cls = res.boxes[i].cls.item()

        img_height, img_width = res.boxes[i].orig_shape

        normalized_coordinates = get_normalized_coordinates(res.boxes[i].xyxy, img_height, img_width)

        if cls == 0:
            info_dict['class'] = names[0]
        elif cls == 1:
            info_dict['class'] = names[1]
        elif cls == 2:
            info_dict['class'] = names[2]
        elif cls == 3:
            info_dict['class'] = names[3]

        info_dict['coordinates'] = normalized_coordinates
        info_dict['left'], info_dict['top'] = normalized_coordinates[0]*100, normalized_coordinates[1]*100
        info_dict['img_height'], info_dict['img_width'] = img_height, img_width
        info_dict['elem_width'] = (normalized_coordinates[2] - normalized_coordinates[0]) * 100
        info_dict['elem_height'] = (normalized_coordinates[3] - normalized_coordinates[1]) * 100
        
        is_text_region = False

        if info_dict['class']=="paragraph" or info_dict['class']=="text_box":
            is_text_region = True
            global_coordinates = get_original_coordinates(normalized_coordinates, info_dict["img_width"], info_dict["img_height"])
            global_coordinates_all_box.append(global_coordinates)
            x_min, y_min, x_max, y_max = global_coordinates

            cropped_text_region = image[y_min:y_max, x_min:x_max]
            # print(y_min,y_max, x_min,x_max)
            # cv2_imshow(cropped_text_region)
            '''uncomment to enable padding'''
            cropped_text_region, padding_amount_box  = top_bottom_padding(
            # cropped_text_region  = top_bottom_padding(
                cropped_text_region, 
                # image.shape, 
                # global_coordinates
            )
            from datetime import datetime
            timestr = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
            cv2.imwrite(f'./dla_inference/{file_name}_cropped_{timestr}.jpg', cropped_text_region)
            cropped_text_region_all_boxes.append(cropped_text_region)
            padding_amount_all_boxes.append(padding_amount_box)
            # print("padding_amount", padding_amount)

        elif info_dict['class']=="image":
            x_min, y_min, x_max, y_max = get_original_coordinates(normalized_coordinates, info_dict["img_width"], info_dict["img_height"])
            cropped_image_region = image[y_min:y_max, x_min:x_max]
            src = f'{img_src_save_directory}/{file_name}_{i}.{extension}'
            info_dict['img_src'] = f'./img/{file_name}_{i}.{extension}'
            # cv2.imwrite(f'html/img/{file_name}_{i}.{extension}', cropped_image_region)
            cv2.imwrite(src, cropped_image_region)
            
        if not is_text_region:
            global_coordinates_all_box.append(None)
            cropped_text_region_all_boxes.append(None)
            padding_amount_all_boxes.append(None)
            


        region_of_interests.append(info_dict)
        
    # https://github.com/BengaliAI/ocr/blob/97f1aa4d5a01629b286e46a47b3287d65a265ce6/runtime%20optimization/Reconstruction_Compiled-line-word-batched_v1v2merged.py#L1028
    # handling overlapping text regions
    # What is the dla box priority?
    discard_elements = []
    for i, element in enumerate(region_of_interests):
        bb1 = box(element['coordinates'][0], element['coordinates'][1], element['coordinates'][2], element['coordinates'][3])

        for j, other_element in enumerate(region_of_interests):
            if j > i:
                bb2 = box(other_element['coordinates'][0], other_element['coordinates'][1], other_element['coordinates'][2], other_element['coordinates'][3])
                intersection = bb1.intersection(bb2).area

                iou = intersection / bb2.area
                # print(i, iou)
                if iou > 0.5:
                    if j not in discard_elements: discard_elements.append(j)

                # if bb1.area < bb2.area:
                #     iou = intersection / bb1.area
                #     if iou > 0.5:
                #         if i not in discard_elements: discard_elements.append(i)
                # else:
                #     iou = intersection / bb2.area
                #     if iou > 0.5:
                #         if j not in discard_elements: discard_elements.append(j)

    # print("discard", discard_elements)

    items_deleted = 0
    for index in discard_elements:
        # del region_of_interests[index - items_deleted]
        # del cropped_text_region_all_boxes[index - items_deleted]
        # del padding_amount_all_boxes[index - items_deleted]
        # del global_coordinates_all_box[index - items_deleted]
        region_of_interests[index] = None
        cropped_text_region_all_boxes[index] = None
        padding_amount_all_boxes[index] = None
        global_coordinates_all_box[index] = None

        # cropped_text_region_all_boxes = []
        # padding_amount_all_boxes = []
        # global_coordinates_all_box = []
        
        items_deleted += 1
 
    # print("global_coordinates_all_box", global_coordinates_all_box); exit()
    region_of_interests = list(filter(lambda x: x is not None, region_of_interests))
    cropped_text_region_all_boxes = list(filter(lambda x: x is not None, cropped_text_region_all_boxes))
    padding_amount_all_boxes = list(filter(lambda x: x is not None, padding_amount_all_boxes))
    global_coordinates_all_box = list(filter(lambda x: x is not None, global_coordinates_all_box))
    
    # viz_all_dlabox(image, global_coordinates_all_box, "dla_gc")
    
    dur = time.time() - start
    runtime['dla_p'] = dur
    
    

    # For standalone OCR
    import copy
    cropped_text_region_all_boxes_for_std_ocr = copy.deepcopy(cropped_text_region_all_boxes)
    global_coordinates_all_box_for_std_ocr = copy.deepcopy(global_coordinates_all_box)
    padding_amount_all_boxes_for_std_ocr = copy.deepcopy(padding_amount_all_boxes)
    # word_crops_all_dlabox = []
    # word_boxes_all_dlabox = []
    # for dlabox_crop in cropped_text_region_all_boxes_for_std_ocr:
    #     all_result_words_this_dlabox = word_segmentation(dlabox_crop)
    #     # all_words_all_dlabox.append(all_words_this_dlabox)
    #     if len(all_result_words_this_dlabox) != 0:
    #         word_crops_this_dlabox, word_boxes_this_dlabox = crop_word_regions(dlabox_crop, [all_result_words_this_dlabox])

    #         word_crops_all_dlabox.append(word_crops_this_dlabox)
    #         word_boxes_all_dlabox.append(word_boxes_this_dlabox)

    
    # import pdb; pdb.set_trace()



    
    global pool1
    # cropped_text_region_all_boxes shape: len(boxes) para/text = 91
    start = time.time()
    print('Num text regions', len(cropped_text_region_all_boxes))
    result_line_all = pool1.map(line_predict, cropped_text_region_all_boxes)
    # result_line_all: co-ordiates with respect to padded cropped textregions
    
    dur = time.time() - start
    runtime['line'] = dur
    print("Line parallel inference time", dur)
    
    start = time.time()

    num_of_lines_per_box = []
    num_of_text_regions = len(result_line_all)
    for i in range (num_of_text_regions):
        # print('num of lines in each para', len(result_line_all[i]))
        num_of_lines_per_box.append(len(result_line_all[i]))

    # print(num_of_lines_per_box)

    
    # result_line_all = for each box, all lines / list[list]
    
    # import pdb; pdb.set_trace()
    
    cropped_line_region_allboxes_all_lines = []
    sorted_line_coordinates_allboxes = []
    line_counts_all_boxes = [0]
    line_coordinates_gc_all_boxes = []
    counter = 0
    # TODO: info dict inplace update
    for info_dict, result_line, cropped_text_region, padding_amount_box, global_coordinates_box in zip(region_of_interests, result_line_all, cropped_text_region_all_boxes, padding_amount_all_boxes, global_coordinates_all_box):
        line_counter = 0
        # result_line = line_segmentation(cropped_text_region)
        # import pdb; pdb.set_trace()
        # global pool
        # result_line = pool.map(process_predict, [cropped_text_region])
        # print("result_line", result_line)

        line_coordinates, line_coordinates_gc = get_coordinates_from_segmentation_fd(result_line, padding_amount_box, global_coordinates_box)
        # line_coordinates_gc_all_boxes.append(line_coordinates_gc) # TODO: remove as only for viz
        # line_coordinates_gc: x_min, y_min, x_max, y_max
        # print("line_coordinates", line_coordinates)

        '''sort line coordinates based on y_min'''
        # Line ordering
        sorted_line_coordinates = sorted(line_coordinates, key = lambda x: x[1])
        sorted_line_coordinates_gc = sorted(line_coordinates_gc, key = lambda x: x[1]) # TODO: check?
        # print("sorted_line_coordinates", sorted_line_coordinates)
        # line_coordinates_gc_all_boxes.append(sorted_line_coordinates_gc) # TODO: remove as only for viz

        '''uncomment to enable vertical dilation for line segments'''
        sorted_line_coordinates, sorted_vertical_dilation_all_lines = line_vertical_dilation(sorted_line_coordinates, cropped_text_region.shape[0], rate=0.1)

        sorted_line_coordinates, sorted_horizontal_dilation_all_lines = line_horizontal_dilation(sorted_line_coordinates, cropped_text_region.shape[1], rate=0.3)

        # print("sorted_line_coordinates", sorted_line_coordinates)

        # print("sorted_line_coordinates after dilation", sorted_line_coordinates)
        for idx, dil in enumerate(sorted_vertical_dilation_all_lines):
            sorted_line_coordinates_gc[idx][1] -= dil[0]
            sorted_line_coordinates_gc[idx][3] += dil[1]
        
        for idx, dil in enumerate(sorted_horizontal_dilation_all_lines):
            sorted_line_coordinates_gc[idx][0] -= dil[0]
            sorted_line_coordinates_gc[idx][2] += dil[1]
        
        # print("sorted_line_coordinates_gc", sorted_line_coordinates_gc)

        line_coordinates_gc_all_boxes.append(sorted_line_coordinates_gc) # TODO: remove as only for viz

        text = []

        total_word_count = 0
        
        sorted_line_coordinates_allboxes.append(sorted_line_coordinates)
        
        
        for i in range(len(sorted_line_coordinates)):
            # cropped_text_region is still wrt paddeded DLA box
            # sorted_line_coordinates is dilated wrt cropped_text_region
            cropped_line_region = cropped_text_region[sorted_line_coordinates[i][1]:sorted_line_coordinates[i][3],
                                                        sorted_line_coordinates[i][0]:sorted_line_coordinates[i][2]]
            
            # if i==0 or i==len(sorted_line_coordinates)-1:
            #     cropped_line_region = top_bottom_padding(cropped_line_region, ratio=1.2)


            if len(sorted_line_coordinates) == 1:
                info_dict['single-line'] = True
                info_dict['elem_height'] *= 1.5

            if len(cropped_line_region) != 0:
                # cv2_imshow(cropped_line_region)
                global cropped_line_region_save_path
                # print("cv empty ***********", cropped_line_region, len(cropped_line_region))
                if LOG_ALL:
                    cv2.imwrite(os.path.join(cropped_line_region_save_path, f"{counter}.jpg"), cropped_line_region)
                counter += 1
            # print("cropped_line_region", cropped_line_region)


            if len(cropped_line_region) != 0:
                # print("*********cropped_line_region*** size: ", line_counter)
                cropped_line_region_allboxes_all_lines.append(cropped_line_region)
                
                line_counter += 1
                
                
                # result_word = word_segmentation(cropped_line_region)
        line_counts_all_boxes.append(line_counts_all_boxes[-1] + line_counter)
    
    # viz_wordboxes_in_all_dlabox(image, global_coordinates_all_box, line_coordinates_gc_all_boxes, "dla_line_dilated_hor150pc")
    
    dur = time.time() - start
    runtime['line_p'] = dur

    global pool2
    print('Num word regions', len(cropped_line_region_allboxes_all_lines))
    start = time.time()
    result_words_allbox_alllines = pool2.map(word_predict, cropped_line_region_allboxes_all_lines)
    # result_words_allbox_alllines = pool2.map(word_batch_predict, [cropped_line_region_allboxes_all_lines])[0]
    # print(result_words_allbox_alllines)
    dur = time.time() - start
    runtime['word'] = dur
    print("Word parallel inference time", dur)

    # import pdb; pdb.set_trace()

    # print('result_words_allbox_alllines', result_words_allbox_alllines, len(result_words_allbox_alllines))
    
    __start = time.time()
    num_of_lines = len(result_words_allbox_alllines)
    num_of_words_per_line = []
    for i in range(num_of_lines):
        num_of_words_per_line.append(len(result_words_allbox_alllines[i]))

    # print('num_of_words_per_line', num_of_words_per_line)
    
    # 9, 10, 11
    # 0, 9, 19, 30

    word_gc_all_box_all_lines = []
    
    for idx in range(len(sorted_line_coordinates_allboxes)):
        start = line_counts_all_boxes[idx]
        end = line_counts_all_boxes[idx+1]
        line_coordinates_gc_this_box = line_coordinates_gc_all_boxes[idx]
        word_gc_this_box_all_lines = []
        
        for cropped_line_region, result_word, line_coordinates_gc in zip(cropped_line_region_allboxes_all_lines[start:end], result_words_allbox_alllines[start:end], line_coordinates_gc_this_box):
            word_gc_this_box_this_line = []
        
            # import pdb; pdb.set_trace()
            # print("result_word", result_word)

            # word_coordinates = get_coordinates_from_segmentation(result_word)
            # print("word_coordinates", word_coordinates)

            # sort words based on x_min
            sorted_result_word = sorted(result_word, key = lambda x: x[0])
            # print("sorted_result_word", sorted_result_word)
            
            
            for idx, word in enumerate(sorted_result_word):
                # print(word)
                # [408, 8, 571, 11, 571, 57, 408, 55]
                x_min, y_min, x_max, y_max = min(word[0],word[6]), min(word[1],word[3]), max(word[2], word[4]), max(word[5], word[7]) 
                sorted_result_word[idx] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                # print(sorted_result_word[idx])

            if len(sorted_result_word) != 0:

                crops, boxes = crop_word_regions(cropped_line_region, [sorted_result_word])
                crops, boxes = word_horizontal_dilation(boxes, cropped_line_region, vertical_dilation = True, ratio = 1.1)
                # import pdb; pdb.set_trace()
                # TODO: fix 
                total_word_count += len(crops)
                word_crops.append(crops)
                word_boxes.append(boxes)

                for word_box in boxes:
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] = word_box
                    word_gc_this_box_this_line_this_word = [
                        line_coordinates_gc[0] + x_min,
                        line_coordinates_gc[1] + y_min,
                        line_coordinates_gc[0] + x_max,
                        line_coordinates_gc[1] + y_max,
                    ]

                    word_gc_this_box_this_line.append(word_gc_this_box_this_line_this_word)
            
            word_gc_this_box_all_lines.append(word_gc_this_box_this_line)
        
        word_gc_all_box_all_lines.append(word_gc_this_box_all_lines)
        # return in x_min, y_min, x_max, y_max formate
                
                 

        num_of_words.append(total_word_count)
    # print('len(word_gc_all_box_all_lines)', len(word_gc_all_box_all_lines))   
    # import pdb; pdb.set_trace()
    word_crops = merge_image_arrays(word_crops)
    word_boxes = merge_image_arrays(word_boxes)
    
    dur = time.time() - __start
    runtime['word_p'] = dur
    
    print("word_crops", len(word_crops))
    print("word_boxes", len(word_boxes))
    
    # viz_wordboxes_in_all_dlabox_all_words(image, global_coordinates_all_box, line_coordinates_gc_all_boxes, word_gc_all_box_all_lines, "dla_line_dilated_all_blw")

    if LOG_ALL:
        for idx, crop in enumerate(word_crops):
            cv2.imwrite(os.path.join(cropped_word_regions_save_path, f"{idx}.jpg"), crop)


    rec_time = time.time()
    # texts = recognize_word(word_crops)
    texts = pool3.map(recognize_word, [word_crops])[0]
    # print('texts', texts)
    dur = time.time() - rec_time
    print('length of words', len(texts))
    print("Recognition time", dur)
    runtime['rec'] = dur


    # For standalone OCR
    # standalone_OCR_required = False
    
    
    
    if standalone_OCR_required:
        
        # cropped_text_region_all_boxes_for_std_ocr_merged = []
        # for list_crop in cropped_text_region_all_boxes_for_std_ocr:
        #     cropped_text_region_all_boxes_for_std_ocr_merged += list_crop
        
        # import pdb; pdb.set_trace()
        result_words_allbox_merged = pool2.map(word_predict, cropped_text_region_all_boxes_for_std_ocr)
        
        result_words_allbox = result_words_allbox_merged
        
        # start = 0
        # result_words_allbox = []
        # for list_crop in cropped_text_region_all_boxes_for_std_ocr:
        #     length = len(list_crop)
        #     result_words_allbox.append(result_words_allbox_merged[start:start+length])
        #     start += length
            

        # import pdb; pdb.set_trace()
        
        all_crops_all_dlabox = []
        all_bboxes_all_dla_box = []
        all_bboxes_all_dla_box_gc = []
        list_list_word_gc = []
        
        # import pdb; pdb.set_trace()

        for box_idx, (result_word, padding) in enumerate(
            zip(result_words_allbox, padding_amount_all_boxes_for_std_ocr)):
            list_word_gc = []

            box_x_min, box_y_min, box_x_max, box_y_max = global_coordinates_all_box_for_std_ocr[box_idx]

            for word_idx, word in enumerate(result_word):
                x_min, y_min, x_max, y_max = min(word[0],word[6]), min(word[1],word[3]), max(word[2], word[4]), max(word[5], word[7]) 
                result_word[word_idx] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                
                # import pdb; pdb.set_trace()

                # x_min = x_min + box_x_min
                # y_min = y_min + box_y_min
                # x_max = x_max + box_x_min
                # y_max = y_max + box_y_min

                # all_bboxes_all_dla_box_gc.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
                

            # import pdb; pdb.set_trace()
            # if len(result_word) != 0:
            if 1:
                # import pdb; pdb.set_trace()

                crops, boxes = crop_word_regions(cropped_text_region_all_boxes_for_std_ocr[box_idx], [result_word])
                # import pdb; pdb.set_trace()
                # crops, boxes = word_horizontal_dilation(boxes, cropped_text_region_all_boxes_for_std_ocr[box_idx], vertical_dilation = True, ratio = 1.1)
                # import pdb; pdb.set_trace()
                all_crops_all_dlabox.append(crops)
                all_bboxes_all_dla_box.append(boxes)

                for word_box in boxes:
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] = word_box
                    word_gc_this_box_this_line_this_word = [
                        box_x_min + x_min,
                        box_y_min + y_min - padding[0],
                        # x_max - x_min,
                        # y_max - y_min
                        box_x_min + x_max,
                        box_y_min + y_max  - padding[1]
                    ]
                    
                    word_gc_this_box_this_line_this_word_dump = [
                        word_gc_this_box_this_line_this_word[0],
                        word_gc_this_box_this_line_this_word[1],
                        word_gc_this_box_this_line_this_word[2] - word_gc_this_box_this_line_this_word[0],
                        word_gc_this_box_this_line_this_word[3] - word_gc_this_box_this_line_this_word[1],
                    ]
                    
                    all_bboxes_all_dla_box_gc.append(word_gc_this_box_this_line_this_word_dump)
                    list_word_gc.append(word_gc_this_box_this_line_this_word)
                    
                    # if type(word_gc_this_box_this_line_this_word) == type('1.0'):
                    #     import pdb; pdb.set_trace()
                    
                    # all_bboxes_all_dla_box_gc.append(four_tuple_to_xyxy_box(word_gc_this_box_this_line_this_word))
                    
            
                gt_text_bbox = [box_x_min, box_y_min, box_x_max, box_y_max]
                # import pdb; pdb.set_trace()

                viz_wordboxes_in_all_dlabox(image, [gt_text_bbox], [list_word_gc], prefix=f"_std{box_idx}", is_input_xyxy=True)
        # import pdb; pdb.set_trace()
            list_list_word_gc.append(list_word_gc)

        all_crops_all_dlabox = merge_image_arrays(all_crops_all_dlabox)
        all_bboxes_all_dla_box = merge_image_arrays(all_bboxes_all_dla_box)

        all_texts_all_dlabox = pool3.map(recognize_word, [all_crops_all_dlabox])[0]
        
        standalone_all_pred_word_bboxes_file = os.path.join(base_path, standalone_all_pred_word_bboxes_dir, file_name+standalone_all_pred_word_bboxes_suffix)
        standalone_all_pred_word_texts_file = os.path.join(base_path, standalone_all_pred_word_texts_dir, file_name+standalone_all_pred_word_texts_suffix)
        
        Path(standalone_all_pred_word_bboxes_file).parent.mkdir(exist_ok=True, parents=True)
        Path(standalone_all_pred_word_texts_file).parent.mkdir(exist_ok=True, parents=True)

        assert len(all_bboxes_all_dla_box_gc) == len(all_texts_all_dlabox), 'Length of all_bboxes_all_dla_box_gc and all_texts_all_dlabox not equal'
        pickle_dump(all_bboxes_all_dla_box_gc, standalone_all_pred_word_bboxes_file)
        pickle_dump(all_texts_all_dlabox, standalone_all_pred_word_texts_file)
    # import pdb; pdb.set_trace()
    
    
    start = time.time()

    paragraph_wise_word_boxes = []
    paragraph_wise_recognized_text = []

    text_region_count = -1
    start_line_idx = start_txt_idx = 0
    

    for elem in region_of_interests:
        if elem['class'] == 'paragraph' or elem['class'] == 'text_box':

            text_region_count += 1
            end_line_idx = start_line_idx + num_of_lines_per_box[text_region_count]
            end_txt_idx = start_txt_idx + sum(num_of_words_per_line[start_line_idx:end_line_idx])

            start_line_idx = end_line_idx

            if end_txt_idx > len(texts):
                end_txt_idx = len(texts)

            # print(start_txt_idx, end_txt_idx)

            paragraph_wise_recognized_text.append(texts[start_txt_idx:end_txt_idx])
            text = ' '.join(texts[start_txt_idx:end_txt_idx])
            paragraph_wise_word_boxes.append(word_boxes[start_txt_idx:end_txt_idx])
            elem['text'] = text
            start_txt_idx = end_txt_idx

    dur = time.time() - start
    runtime['rec_p'] = dur
    # print("len(paragraph_wise_word_boxes)", len(paragraph_wise_word_boxes))
    # print("len(paragraph_wise_recognized_text)", paragraph_wise_recognized_text)

    '''# Now non delayed postprocess
    # handling overlapping text regions
    discard_elements = []
    for i, element in enumerate(region_of_interests):
        bb1 = box(element['coordinates'][0], element['coordinates'][1], element['coordinates'][2], element['coordinates'][3])

        for j, other_element in enumerate(region_of_interests):
            if j > i:
                bb2 = box(other_element['coordinates'][0], other_element['coordinates'][1], other_element['coordinates'][2], other_element['coordinates'][3])
                intersection = bb1.intersection(bb2).area

                iou = intersection / bb2.area
                # print(i, iou)
                if iou > 0.5:
                    if j not in discard_elements: discard_elements.append(j)

                # if bb1.area < bb2.area:
                #     iou = intersection / bb1.area
                #     if iou > 0.5:
                #         if i not in discard_elements: discard_elements.append(i)
                # else:
                #     iou = intersection / bb2.area
                #     if iou > 0.5:
                #         if j not in discard_elements: discard_elements.append(j)

    # print("discard", discard_elements)

    # assert len(region_of_interests) == len(paragraph_wise_word_boxes), f'{len(region_of_interests)}, {len(paragraph_wise_word_boxes)}Inequal roi and paragraph wise word box'
    roi_text_index_map = {}
    text_count = 0
    for idx, roi in enumerate(region_of_interests):
        if region_of_interests[idx]['class'] in ['paragraph', 'text_box']:
            roi_text_index_map[idx] = text_count
            text_count += 1

    for index in discard_elements:
        if 0 <= (index) < len(region_of_interests):
            
            if region_of_interests[index]['class'] in ['paragraph', 'text_box']:
                region_of_interests[index] = None    
                paragraph_wise_word_boxes[roi_text_index_map[index]] = None
                paragraph_wise_recognized_text[roi_text_index_map[index]] = None
    
    region_of_interests = list(filter(lambda x: x is not None, region_of_interests))
    paragraph_wise_word_boxes = list(filter(lambda x: x is not None, paragraph_wise_word_boxes))
    paragraph_wise_recognized_text = list(filter(lambda x: x is not None, paragraph_wise_recognized_text))
    '''

    # print("len(paragraph_wise_recognized_text)", paragraph_wise_recognized_text)

    file_stem = file_name

    roi_file = os.path.join(base_path, roi_directory, file_stem+roi_suffix)
    predicted_text_file = os.path.join(base_path, predicted_text_directory, file_stem+pred_text_suffix)
    paragraph_wise_predicted_word_box_file =  os.path.join(base_path, paragraph_wise_predicted_word_box_directory, file_stem+paragraph_wise_predicted_word_box_suffix)
    word_gc_all_box_all_lines_file = os.path.join(base_path, word_gc_all_box_all_lines_directory, file_stem+word_gc_all_box_all_lines_suffix)
    # print(word_gc_all_box_all_lines_file)

    pickle_dump(region_of_interests, roi_file)
    pickle_dump(paragraph_wise_recognized_text, predicted_text_file)
    pickle_dump(paragraph_wise_word_boxes, paragraph_wise_predicted_word_box_file)
    pickle_dump(word_gc_all_box_all_lines, word_gc_all_box_all_lines_file)
    

    # assert len(region_of_interests) == len(paragraph_wise_word_boxes), 'Inequal roi and paragraph wise word box'

    # print('region_of_interests', region_of_interests)
    return region_of_interests, runtime

# %% [markdown]
# Generate html file

# %%
def reconstruct(directory, img_src_save_dir, standalone_OCR_required):

    import pandas as pd
    from tqdm import tqdm

    im_names = []
    full_runtimes = []


    # Iterate over all files in the directory
    for idx, file_name in tqdm(enumerate(os.listdir(directory))):
        if os.path.isfile(os.path.join(directory, file_name)):
            if file_name.endswith('.csv'): continue
        # print(file_name)
        if os.path.isfile(os.path.join(directory, file_name)):
        # if os.path.isfile(os.path.join(directory, file_name))  \
            # and file_name in ['d3f09723-2b93-4c28-8166-c95daa11de79.png']:
            # and file_name in ["0b0e7f70-1225-4ed5-b5e2-9cc5fb9795a6.png"]:
            # , '0aa9adfb-378e-457a-8e29-f9e75f9611ad.png']:

            file_path = directory + "/" + file_name
            im_names.append(file_name)


            global cropped_word_regions_save_path, cropped_line_region_save_path

            cropped_line_region_save_path = os.path.join(cropped_line_region_save_path_base, file_name.split('.')[0])
            cropped_word_regions_save_path = os.path.join(cropped_word_regions_save_path_base, file_name.split('.')[0])

            os.makedirs(cropped_line_region_save_path, exist_ok=True)
            os.makedirs(cropped_word_regions_save_path, exist_ok=True)

            print("----------------------------------------------------------------------------")
            print("File name:", file_name)

            start_time = time.time()
            roi, runtime = run_yolo_model(model_weight, file_path, file_name, img_src_save_dir, standalone_OCR_required)

            import pickle

            with open('roi.pkl', 'wb') as file:
                pickle.dump(roi, file)


            full_run_time = round(time.time() - start_time, 2)
            
            print("Execution Time for Layout Prediction and Text Recognition:", full_run_time, "seconds")

            start_time = time.time()
            generate_html(roi, file_name)
            html_time = round(time.time() - start_time, 2)
            runtime['html'] = html_time
            
            print("Execution Time for Reconstruction:", html_time, "seconds")
            print("----------------------------------------------------------------------------")
            # break

            # full_runtimes.append(full_run_time+html_time)
            full_runtimes.append(sum(runtime.values()))
            print(runtime)

            print("Avg runtime: ", np.array(full_runtimes).mean())
            pd.DataFrame({'img': im_names, 'full_pipeline': full_runtimes}).to_csv(log_csv_file, index=None)
    

def load_bocr():
    global rec
    rec = BanglaOCR(ONNX_PATH)




def line_predict(image_line):
  # predict ppocr result
  # im = cv2.imread(image)
  result = det_model.predict(image_line)
  return result.boxes


def word_predict(image_word):
  # predict ppocr result
  # im = cv2.imread(image)
  result = det_model2.predict(image_word)
  # print(result)
  return result.boxes


def word_batch_predict(image_word):
  # predict ppocr result
  # im = cv2.imread(image)
  result = det_model2.batch_predict(image_word)
  result = [r.boxes for r in result]
#   print(result)
  return result

from copy import deepcopy
im_path = 'bus.jpg'
# im_path = '/mnt/hdd/jawaril/dataset/icmla23/badlad13k_categorized_50sampleseach/ground_truth/img_to_be_reconstructed/ekal1.jpg'
# im_path = '/mnt/hdd/jawaril/dataset/icmla23/badlad13k_categorized_50sampleseach/ground_truth/img_to_be_reconstructed/two_col.png'
# im_path = '/media/shayekh/Data/_learn_/ocr23/img_to_be_reconstructed/dc (1).png'
imgs_list = []
im = cv2.imread(im_path)

for i in range(100):
  imgs_list.append(deepcopy(im))
  

if __name__ == '__main__':

    
    # %%
    # test_image_directory = 'img_to_be_reconstructed'
    # test_image_directory = 'eval_imgs218'
    test_image_directory = Path(image_dir).name
    
    # test_image_directory = 'eval_imgs_paperedge_local_it1'
    img_src_save_dir = f'./recon_batch_{test_image_directory}/html/img'
    cropped_line_region_save_path_base = f"./cropped_line_region_{test_image_directory}"
    cropped_word_regions_save_path_base = f"./cropped_word_regions_{test_image_directory}"
        # "paragraph_wise_predicted_word_box"
        # "roi"
    global log_csv_file 
    process_num1 = 1 # Optimal: 2
    process_num2 = 1 # Optimal: 3
    process_num1 = 2 # Optimal: 2
    # process_num2 = 3 # Optimal: 3
    process_num2 = 2 # Optimal: 3

    log_csv_file = test_image_directory+f'_{preproc_suffix}_full_runtime_{process_num1}_{process_num2}_160.csv'
    
    
    # process_num1 = 2
    # process_num2 = 2
    pool0 = Pool(
        1,
        initializer=load_yolo,
    )
    
    
    pool1 = Pool(
        process_num1,
        initializer=load_model,
        # initargs=()
      )
    pool2 = Pool(
        process_num2,
        initializer=load_model2,
        # initargs=()
      )
    pool3 = Pool(
        1,
        initializer=load_bocr
    )
    # with Pool(
    #     process_num,
    #     initializer=load_model,
    #     # initargs=()
    #   ) as pool:
    if 1:
        # results = pool3.map(recognize_word, [imgs_list[:process_num]])[0]
        results = pool0.map(dla_predict, [imgs_list[0]])
        results = pool1.map(line_predict, imgs_list[:process_num1])
        results = pool2.map(word_predict, imgs_list[:process_num2])
        results = pool3.map(recognize_word, [[imgs_list[0]]])[0]
    
    # exit()

    # os.makedirs(img_src_save_dir, exist_ok=True, parent=True)
        Path(img_src_save_dir).mkdir(parents=True, exist_ok=True)
        # os.makedirs(cropped_line_region_save_path, exist_ok=True)
        # os.makedirs(cropped_word_regions_save_path, exist_ok=True)
        reconstruct(image_dir, img_src_save_dir, 
                    # standalone_OCR_required= True
                    standalone_OCR_required= False
                    )
        
    # pool1.join()
    # pool2.join()
    # pool3.join()
    pool0.close()
    pool1.close()
    pool2.close()
    pool3.close()


