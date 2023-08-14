# base_path = '/media/shayekh/Data/_learn_/ocr23/full_pipeline/Evaluation_Metric' # shayekh
# image_dir = r'../../eval_imgs218' # shayekh

base_path = '/mnt/hdd/jawaril/full_pipeline/Evaluation_Metric'
# base_path = r'D:\BADLAD\__RECONSTRUCTION\Evaluation_Metric'
# image_dir = r'data/image'
image_dir = r'data/eval_218/images'

# word_annotation_directory = r'data/word_annotation'
word_annotation_directory = r'data/eval_218/word_annotations'

data_domain_directory = r'data/eval_218'

roi_directory = r'data/roi'
paragraph_wise_predicted_word_box_directory = r'data/paragraph_wise_predicted_word_box'
predicted_text_directory = r'data/predicted_text'
cer_csv_directory = r'data/cer_csv'
roi_suffix = '_region_of_interests.pkl'
pred_text_suffix = '_paragraph_wise_recognized_text.pkl'
paragraph_wise_predicted_word_box_suffix = '_paragraph_wise_word_boxes.pkl'
badlad_coco_path = r'data/badlad-test-coco.json'
word_gc_all_box_all_lines_directory = r'data/word_gc_all_box_all_lines'
word_gc_all_box_all_lines_suffix = '_word_gc_all_box_all_lines.pkl'

error_analysis = {
    'f335e15b-83ea-44b7-b11d-29635752166c': 'blurry image, geom/illum/superres issues',
    'f1e2128b-b41a-4936-ae53-fdcf9b24ddb9': 'magagine cover, misses completely',
    'd8838650-303d-47de-9a8c-62a5c7e153e7': 'full table, all missing, table reconstruction is out of scope now',
    'edbdeb19-814b-4c68-94c9-a5cc3da10ec5': '80% table',
    'dbc46bb8-7b5c-48b7-ac6f-ecb074b0e1b4': 'book cover, detected as image',
    'd2114573-a8fa-475e-92ca-fe473906512b': '',
}


import pickle
import cv2
import math
from shapely.geometry import box
import json
from pathlib import Path
import os
from nltk.metrics.distance import edit_distance
import csv
from tqdm import tqdm
import pandas as pd
import copy

# os.makedirs('tmp', exist_ok=True)
os.makedirs('viz', exist_ok=True)

def get_original_coordinates(normalized_coordinates, image_width, image_height):
    orig_coordinates = [None]*4

    orig_coordinates[0] = math.floor(normalized_coordinates[0] * image_width)
    orig_coordinates[1] = math.floor(normalized_coordinates[1] * image_height)
    orig_coordinates[2] = math.ceil(normalized_coordinates[2] * image_width)
    orig_coordinates[3] = math.ceil(normalized_coordinates[3] * image_height)

    return orig_coordinates


def change_bbox_format_xywh(all_predicted_text_region_bbox):
    formatted_all_predicted_text_region_bbox = []

    for region in all_predicted_text_region_bbox:
        new_bbox = [region[0], region[1], region[2]-region[0], region[3]-region[1]]
        formatted_all_predicted_text_region_bbox.append(new_bbox)
    
    return formatted_all_predicted_text_region_bbox


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import partial

    

# %% [markdown]
# names = {0: 'paragraph', 1: 'text_box', 2: 'image', 3: 'table'}

def get_all_annotated_word_and_bbox(word_annotation):

    all_annotated_words = []
    all_annotated_words_bbox = []

    for i in range(len(word_annotation['regions'])):

        single_word = word_annotation['regions'][i]
        # if(single_word['shape_attributes']['name']!='polygon'):
        # TODO: consider polygon as well
        if(single_word['shape_attributes']['name']=='rect'):
            x = single_word['shape_attributes']['x']
            y = single_word['shape_attributes']['y']
            width = single_word['shape_attributes']['width']
            height = single_word['shape_attributes']['height']

            all_annotated_words.append(single_word['region_attributes']['name'])
            all_annotated_words_bbox.append([x, y, width, height])

    return all_annotated_words, all_annotated_words_bbox


def get_all_predicted_text_region_box(roi):

    '''returns:
        [x,y,w,h]'''
    
    all_predicted_text_region_bbox = []

    for region in roi:
        if region['class'] in ['paragraph', 'text_box']:
            
            norm_coordinates = region['coordinates']
            x_min, y_min, x_max, y_max = get_original_coordinates(norm_coordinates, region["img_width"], region["img_height"])
            all_predicted_text_region_bbox.append([x_min, y_min, x_max-x_min, y_max-y_min])
    
    return all_predicted_text_region_bbox


def map_predicted_text_region_to_gt_region(all_predicted_text_region_bbox, all_gt_text_region_bbox):
    
    '''
    pred_box_gt_box_mapper = {pred_text_region_id: gt_text_region_id, ....}
    id = index of their respective list
    '''

    pred_box_gt_box_mapper = {}

    intersection_thresh = 0.5 # TODO: Verify/tune
    # intersection_thresh = 0.1 # TODO: Verify/tune
    

    for pred_region_id, pred_region in enumerate(all_predicted_text_region_bbox):
        # pred_region -> [xmin, ymin, xmax, ymax]
        pred_box_gt_box_mapper[pred_region_id] = -1
        pred_region_shapely_box = box(pred_region[0], pred_region[1], pred_region[0]+pred_region[2], pred_region[1]+pred_region[3])

        # print(f"pred_region_shapely_box: {pred_region_shapely_box}")

        # plt.plot(*pred_region_shapely_box.exterior.xy)
        # import pdb; pdb.set_trace()

        max_intersection = 0
        # print(f"pred_region_id {pred_region_id}")
        for gt_region_id, gt_region in enumerate(all_gt_text_region_bbox):
            # gt_region -> [xmin, ymin, w, h]
            gt_region_shapely_box = box(gt_region[0], gt_region[1], gt_region[0]+gt_region[2], gt_region[1]+gt_region[3])
            # import pdb; pdb.set_trace()
            # print(f"pred_region_id {pred_region_id}, gt_region_id {gt_region_id}\npred_region_shapely_box: {pred_region_shapely_box}, \ngt_region_shapely_box: {gt_region_shapely_box}")

            # plt.plot(*pred_region_shapely_box.exterior.xy)
            # print(pred_region_shapely_box.exterior.xy)
            # plt.show()

            # we normalize wrt pred: because we assume a single gt text-region can be broken down into multiple pred text-region, we assume no tie
            # we greedily assign pred to max box: ingore tie for now as dla works good as per human eval
            intersection_over_pred_region = pred_region_shapely_box.intersection(gt_region_shapely_box).area / pred_region_shapely_box.area
            # print(f"intersection_over_pred_region: {intersection_over_pred_region}, max_intersection: {max_intersection}")

            if intersection_over_pred_region > max(max_intersection, intersection_thresh):
                # print(f"intersection_over_pred_region: {intersection_over_pred_region}, max_intersection: {max_intersection}")
                pred_box_gt_box_mapper[pred_region_id] = gt_region_id
                max_intersection = intersection_over_pred_region
            # print(f"intersection_over_pred_region: {intersection_over_pred_region}, max_intersection: {max_intersection}")
        
        # print(f"pred_box_gt_box_mapper {pred_box_gt_box_mapper}")


    return pred_box_gt_box_mapper


def draw_bbox(image, bbox, idx, pred):
    x, y, width, height = bbox
    fig, ax = plt.subplots()
    ax.imshow(image)
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.savefig(f'tmp/ImagewithBoundingBox_{idx}_{"pred" if pred else "gt"}.png')

def viz_pred_box_gt_box_mapper(image, pred_box_gt_box_mapper, all_predicted_text_region_bbox, all_gt_text_region_bbox):
    for idx, (k,v) in enumerate(pred_box_gt_box_mapper.items()):
        pred_bbox = all_predicted_text_region_bbox[k]
        draw_bbox(image, pred_bbox, idx, True)

        gt_bbox = all_gt_text_region_bbox[v]
        draw_bbox(image, gt_bbox, idx, False)
    

def viz_wordboxes_in_all_dlabox(image, all_gt_text_region_bbox, wordboxes_in_all_dlabox):
    for idx, (dla_box, word_boxes) in enumerate(zip(all_gt_text_region_bbox, wordboxes_in_all_dlabox)):
        x, y, width, height = dla_box
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(image)
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for word_box in word_boxes:
            x, y, width, height = word_box
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            # break
        
        plt.savefig(f'tmp/dlabox_{idx}_gt.png', pad_inches=0.2)


'''
Inside, the gt dla-box, find all the gt word-boxes

Assign: dump per dlabox
'''
def extract_wordbox_gt_from_dlabox_gt(all_gt_text_region_bbox, all_annotated_words_bbox, all_annotated_words):
    # print("extract_wordbox_gt_from_dlabox_gt")
    # print(f"all_gt_text_region_bbox: {all_gt_text_region_bbox}, \nall_annotated_words_bbox: {all_annotated_words_bbox}")

    wordboxes_in_all_dlabox = []
    wordtext_in_all_dlabox = []
    # intersection_thresh_wordbox_dlabox = 0.7
    intersection_thresh_wordbox_dlabox = 0.5
    # intersection_thresh_wordbox_dlabox = 0.1
    for dlabox in all_gt_text_region_bbox:
        wordboxes_in_dlabox = []
        wordtext_in_dlabox = []
        dlabox_shapely_box = box(dlabox[0], dlabox[1], dlabox[0]+dlabox[2], dlabox[1]+dlabox[3])

        for idx, wordbox in enumerate(all_annotated_words_bbox):
            wordbox_shapely_box = box(wordbox[0], wordbox[1], wordbox[0]+wordbox[2], wordbox[1]+wordbox[3])
            if wordbox_shapely_box.area < 1e-5: continue

            intersection_over_wordbox = wordbox_shapely_box.intersection(dlabox_shapely_box).area / wordbox_shapely_box.area
            
            # print(f"intersection_over_pred_region: {intersection_over_pred_region}, max_intersection: {max_intersection}")

            if intersection_over_wordbox > intersection_thresh_wordbox_dlabox:
                # print(f"intersection_over_pred_region: {intersection_over_pred_region}, max_intersection: {max_intersection}")
                wordboxes_in_dlabox.append(wordbox)
                wordtext_in_dlabox.append(all_annotated_words[idx])


        wordboxes_in_all_dlabox.append(wordboxes_in_dlabox)
        wordtext_in_all_dlabox.append(wordtext_in_dlabox)

    # print("wordboxes_in_all_dlabox", wordboxes_in_all_dlabox)
    assert len(wordtext_in_all_dlabox) == len(wordtext_in_all_dlabox)
    return wordboxes_in_all_dlabox, wordtext_in_all_dlabox



from pprint import pprint

def four_tuple_to_xyhw_box(four_tuple_box):
    '''
    [[203.,   8.], [255.,   8.], [255.,  25.], [203.,  25.]] => [x, y, h, w]

    '''
    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] = four_tuple_box

    return list(map(int, [x_min, y_min, x_max - x_min, y_max - y_min]))


def evaluate_cer_lev(gt_word, pred_word):
    # gt_word = gt_word.strip()
    if len(gt_word)==0:
        lev = len(pred_word)
        cer = 1.0

    elif len(pred_word)==0:
        lev = len(gt_word)
        cer = 1.0

    else:
        lev = edit_distance(gt_word, pred_word)
        # cer = lev/len(gt_word)
        # If diving by len(gt_word) may lead to ERD > 1.00 => take max by ICDAR'19
        cer = lev/max(len(gt_word), len(pred_word))

    return cer, lev


def store_in_csv(cer_csv_directory, file_stem, all_gt_word_pred_word_scores, avg_cer, avg_acc, avg_precision, avg_f1, avg_csv_file_path):
    csv_file_path = os.path.join(cer_csv_directory, file_stem+'.csv')
    try:
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        csv_file = open(csv_file_path, mode='w', encoding='utf-8')
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(['dla_box_idx', 'gt_word_idx', 'gt_word', 'pred_word_idx', 'pred_word', 'cer', 'levenshtein distance'])
        all_gt_word_pred_word_scores_sorted_by_dla_id = sorted(all_gt_word_pred_word_scores, key = lambda x: x[0])
        csv_writer.writerows(all_gt_word_pred_word_scores_sorted_by_dla_id)
        csv_file.close()
    except Exception as e:
        print(f"An error occurred: {e}")

    avg_csv_file = open(avg_csv_file_path, mode='a', encoding='utf-8')
    csv_writer = csv.writer(avg_csv_file)
    csv_writer.writerow([file_stem, avg_cer, avg_acc])
    global reconstruction_v1_metric_df
    # reconstruction_v1_metric_df = reconstruction_v1_metric_df.append(
    reconstruction_v1_metric_df.loc[len(reconstruction_v1_metric_df)] = \
        {
            'file_stem': file_stem, 'average_NED': avg_cer, 
            'average_accuracy': avg_acc, 
            'average_precision': avg_precision, 
            'average_f1': avg_f1, 
        }
    
    reconstruction_v1_metric_df.to_csv(reconstruction_v1_metric_file_path, index=None)



def xyxy_to_xywh(data_format):
    return [data_format[0], data_format[1], data_format[2] - data_format[0], data_format[3] - data_format[1]]
    

def xywh_to_xyxy(data_format):
    return [data_format[0], data_format[1], data_format[2] + data_format[0], data_format[3] + data_format[1]]


bn_ocr_vocab = [
                # "blank",
                "!","\"","#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","।",
                        "ঁ","ং","ঃ","অ","আ","ই","ঈ","উ","ঊ","ঋ","এ","ঐ","ও","ঔ",
                        "ক","খ","গ","ঘ","ঙ","চ","ছ","জ","ঝ","ঞ","ট","ঠ","ড","ঢ","ণ","ত","থ","দ","ধ","ন",
                        "প","ফ","ব","ভ","ম","য","র","ল","শ","ষ","স","হ",
                        "া","ি","ী","ু","ূ","ৃ","ে","ৈ","ো","ৌ","্",
                        "ৎ","ড়","ঢ়","য়","০","১","২","৩","৪","৫","৬","৭","৮","৯",
                        # "‍","sep","pad"
                        ]

ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
digits = '0123456789'

englist_vocab = ascii_letters + digits

from bnunicodenormalizer import Normalizer
NORM=Normalizer()

def gt_word_text_filter(gt_word_text):
    # remove start/end spaces,newlines,tabs
    gt_word_text = gt_word_text.strip()
    
    for ch in gt_word_text:
        # if ch not in bn_ocr_vocab:
        if ch in englist_vocab:
            return False, gt_word_text
    
    gt_word_text_frags = gt_word_text.split(' ')
    gt_word_text_frags = [NORM(frag)["normalized"] for frag in gt_word_text_frags]
    # import pdb; pdb.set_trace()
    gt_word_text_frags = list(filter(lambda x: x is not None, gt_word_text_frags))
    gt_word_text = ' '.join(gt_word_text_frags)
    

    
    return True, gt_word_text
    

def reconstruction_v1_avg_cer(
    image, pred_text, roi, word_annotation, 
    all_gt_text_region_bbox, #paragraph_wise_predicted_word_box, 
    word_gc_all_box_all_lines, file_stem, cer_csv_directory, avg_csv_file_path):
    # import pdb; pdb.set_trace()
    assert len(pred_text) == len(word_gc_all_box_all_lines)
    # print(len(word_gc_all_box_all_lines[0][0][0]))
    # print('word_gc_all_box_all_lines', len(word_gc_all_box_all_lines))
    
    word_gc_all_box_all_words = []
    for word_gc_this_box_all_lines in word_gc_all_box_all_lines:
        word_gc_this_box_all_words = []
        for word_gc_this_box_this_line in word_gc_this_box_all_lines:
            word_gc_this_box_all_words += word_gc_this_box_this_line
        
        # Convert [x_min, y_min, x_max, y_max] => [x_min, y_min, w, h]
        word_gc_this_box_all_words = list(map(
            xyxy_to_xywh,
            word_gc_this_box_all_words))
        
            
        word_gc_all_box_all_words.append(word_gc_this_box_all_words)

    
    paragraph_wise_predicted_word_box = word_gc_all_box_all_words

    
        

    '''
    all_annotated_words
    all_annotated_words_bbox

    all_predicted_text_region_bbox
    all_gt_text_region_bbox

    pred_text -> paragraph wise predicted text

    '''
    all_annotated_words, all_annotated_words_bbox = get_all_annotated_word_and_bbox(word_annotation)
    # import pdb; pdb.set_trace()
    all_predicted_text_region_bbox = get_all_predicted_text_region_box(roi)

    deepcopy_all_gt_text_region_bbox = copy.deepcopy(all_gt_text_region_bbox)
    deepcopy_all_predicted_text_region_bbox = copy.deepcopy(all_predicted_text_region_bbox)
    
    pred_dlabox_to_gt_dlabox_mapper = map_predicted_text_region_to_gt_region(all_predicted_text_region_bbox, all_gt_text_region_bbox)

    deepcopy_pred_dlabox_to_gt_dlabox_mapper = copy.deepcopy(pred_dlabox_to_gt_dlabox_mapper)

    # viz_pred_box_gt_box_mapper(image, pred_dlabox_to_gt_dlabox_mapper, all_predicted_text_region_bbox, all_gt_text_region_bbox)
    # print('pred_dlabox_to_gt_dlabox_mapper', pred_dlabox_to_gt_dlabox_mapper)

    wordboxes_in_all_gt_dlabox, wordtext_in_all_gt_dlabox = extract_wordbox_gt_from_dlabox_gt(all_gt_text_region_bbox, all_annotated_words_bbox, all_annotated_words)
    # wordboxes_in_all_gt_dlabox ??? only 50 words out of 350???? fix extract_wordbox_gt_from_dlabox_gt
    # import pdb; pdb.set_trace()
    # viz_wordboxes_in_all_dlabox(image, all_gt_text_region_bbox, wordboxes_in_all_gt_dlabox)
    # print(wordtext_in_all_gt_dlabox)

    # print(pred_dlabox_to_gt_dlabox_mapper)
    # pred_dlabox_to_gt_dlabox_mapper[10] = 0
    gt_dlabox_to_pred_superbox = {}
    # pred to gt => gt to pred conversion
    for pred, gt in pred_dlabox_to_gt_dlabox_mapper.items():
        if gt != -1:
            if gt_dlabox_to_pred_superbox.get(gt, None) is None: 
                gt_dlabox_to_pred_superbox[gt] = [pred]
            else: 
                gt_dlabox_to_pred_superbox[gt] += [pred]

    # Add empty list to the gt dla box for which there is no pred dla box
    # TODO: add none to missing gt
    all_gt_dla_indices = list(range(len(all_gt_text_region_bbox)))
    for gt_dla_index in all_gt_dla_indices:
        if gt_dla_index not in gt_dlabox_to_pred_superbox.keys():
            gt_dlabox_to_pred_superbox[gt_dla_index] = []
    
    # Ignore unalignd pred, unalign gt
    # import pdb; pdb.set_trace()

    deepcopy_gt_dlabox_to_pred_superbox = copy.deepcopy(gt_dlabox_to_pred_superbox)

    # print('deep_copy_gt_dlabox_to_pred_superbox', deep_copy_gt_dlabox_to_pred_superbox)
    # print('gt_dlabox_to_pred_superbox', gt_dlabox_to_pred_superbox)
    gt_dlabox_to_predtext_merged = {}
    gt_dlabox_to_predbox_merged = {}
    for gt, pred_superbox in gt_dlabox_to_pred_superbox.items():

        gt_dlabox_to_predtext_merged[gt] = []
        gt_dlabox_to_predbox_merged[gt] = []
        for superbox in pred_superbox:
            # if gt_dlabox_to_predtext_merged.get(gt, None) is None:
            #     gt_dlabox_to_predtext_merged[gt] = pred_text[superbox]
            #     gt_dlabox_to_predbox_merged[gt] = paragraph_wise_predicted_word_box[superbox]

            # else:
            if 1:
                gt_dlabox_to_predtext_merged[gt] += pred_text[superbox]
                gt_dlabox_to_predbox_merged[gt] += paragraph_wise_predicted_word_box[superbox]
    
    # import pdb; pdb.set_trace()
    
    # print(f"gt_box_to_predtext_merged: {gt_dlabox_to_predtext_merged}")
    # print(f"gt_box_to_predbox_merged: {gt_dlabox_to_predbox_merged.keys()}")

    # Now we know for a gt dlabox: 
    #   all gt word boxes (extract_wordbox_gt_from_dlabox_gt) 
    #   and all pred word boxes (merged) (map_predicted_text_region_to_gt_region, in prev loop)

    # Iterate over all text-region
    
    # pred_without_gt = []
    # gt_without_pred = []

    pred_gt_mapping_all_box = []
    # word_intersection_thresh = 0.7
    word_intersection_thresh = 0.5
    # word_intersection_thresh = 0.95
    # word_intersection_thresh = 0.1
    # import pdb; pdb.set_trace()
    for gt_dla_box_idx, gt_wordboxes in enumerate(wordboxes_in_all_gt_dlabox):
        
        gt_word_texts = wordtext_in_all_gt_dlabox[gt_dla_box_idx]
        # pred pivot
        
        
        pred_gt_mapping_this_box = []
        gt_assigned = []
        # TODO: penalty no pred box for GT dla box
        # if gt_dlabox_to_predbox_merged.get(gt_dla_box_idx, None) is None: 
        #     continue
        for pred_idx, pred_word_box in enumerate(gt_dlabox_to_predbox_merged[gt_dla_box_idx]):
            pred_gt_mapping_this_box.append(-1)
            
            max_intersection = 0
            pred_word_box_xyxy = xywh_to_xyxy(pred_word_box)
            pred_word_box_shapely = box(*pred_word_box_xyxy)
            gt_assigned.append([-1]) # avoid already assign gt
            for gt_idx, (gt_wordbox, gt_wordtext) in enumerate(zip(gt_wordboxes, gt_word_texts)):
                # pass
                gt_wordbox_xyxy = xywh_to_xyxy(gt_wordbox)
                gt_wordbox_shapely = box(*gt_wordbox_xyxy)
                # import pdb; pdb.set_trace()
                # intersection_over_wordbox = pred_word_box_shapely.intersection(gt_wordbox_shapely).area / gt_wordbox_shapely.area
                intersection_over_wordbox = (
                    pred_word_box_shapely.intersection(gt_wordbox_shapely).area / 
                    pred_word_box_shapely.union(gt_wordbox_shapely).area
                )
                # prioritize gt box for normalization
                if intersection_over_wordbox > word_intersection_thresh and \
                    intersection_over_wordbox > max_intersection and \
                    gt_idx not in gt_assigned:
                    
                    max_intersection = intersection_over_wordbox
                    # pred_gt_mapping_this_box[pred_idx] = gt_idx 
                    pred_gt_mapping_this_box[-1] = gt_idx
                    gt_assigned[-1] = gt_idx
        
        pred_gt_mapping_all_box.append(pred_gt_mapping_this_box)
    

    pred_words_count = 0
    for dla_box in pred_gt_mapping_all_box:
        for pred_idx in dla_box:
            if pred_idx != -1:
                pred_words_count += 1

    
    # print(pred_gt_mapping_all_box)
    # import pdb; pdb.set_trace()

    # with gtword as pivot; reverse mapper
    gt_pred_mapping_all_box = []
    for gt_dla_box_idx, gt_wordboxes in enumerate(wordboxes_in_all_gt_dlabox):
        # print(gt_dla_box_idx, gt_wordboxes)
        # pprint(gt_dlabox_to_predbox_merged)
        # gt_word_texts = wordtext_in_all_gt_dlabox[gt_dla_box_idx]
        # pred pivot
        
        gt_pred_mapping_this_box = []
        pred_assigned = []
        # print(len(gt_wordboxes))
        for gt_idx, gt_word_box in enumerate(gt_wordboxes):
            gt_pred_mapping_this_box.append(-1)
            
            # pred_gt_mapping_this_box[pred_idx] = -1
            
            max_intersection = 0
            gt_word_box_xyxy = xywh_to_xyxy(gt_word_box)
            gt_word_box_shapely = box(*gt_word_box_xyxy)
            pred_assigned.append([-1]) # avoid already assign gt
            # print(gt_dlabox_to_predbox_merged)  
            
            # TODO: penalty for missing box
            # if gt_dlabox_to_predbox_merged.get(gt_dla_box_idx, None) is None: 
            #     continue
            
            for pred_idx, pred_word_box in enumerate(gt_dlabox_to_predbox_merged[gt_dla_box_idx]):
                # pass
                # import pdb; pdb.set_trace()
                pred_wordbox_xyxy = xywh_to_xyxy(pred_word_box)
                pred_wordbox_shapely = box(*pred_wordbox_xyxy)
                # import pdb; pdb.set_trace()
                # intersection_over_wordbox = pred_wordbox_shapely.intersection(gt_word_box_shapely).area / gt_word_box_shapely.area
                intersection_over_wordbox = (
                    pred_wordbox_shapely.intersection(gt_word_box_shapely).area / 
                    pred_wordbox_shapely.union(gt_word_box_shapely).area
                )
                # prioritize gt box for normalization
                if intersection_over_wordbox > word_intersection_thresh and \
                    intersection_over_wordbox > max_intersection and \
                    pred_idx not in pred_assigned:
                    max_intersection = intersection_over_wordbox
                    # pred_gt_mapping_this_box[pred_idx] = gt_idx 
                    gt_pred_mapping_this_box[-1] = pred_idx
                    pred_assigned[-1] = pred_idx
            
        gt_pred_mapping_all_box.append(gt_pred_mapping_this_box)


    total_cer = 0
    cer_count = 0
    n_correct = 0
    # total_words = 0
    all_gt_word_pred_word_scores = []
    for gt_dlabox_idx, this_box_all_pred_words_idx in enumerate(gt_pred_mapping_all_box):
        # total_words += len(wordtext_in_all_gt_dlabox[gt_dlabox_idx])
        # import pdb; pdb.set_trace()
        for gt_word_idx, pred_word_idx in enumerate(this_box_all_pred_words_idx):
            gt_word_str = wordtext_in_all_gt_dlabox[gt_dlabox_idx][gt_word_idx]
            # gt_word_str = gt_word_str.strip()
            status, gt_word_str = gt_word_text_filter(gt_word_str)
            if not status:
                # print(f'Ignoring Non-Bengali word {gt_word_str}')
                continue

            # total_words += 1
            
            if pred_word_idx != -1:
                # print(gt_dlabox_idx,pred_word_idx)
                ## TODO: This should not occur ???? fix it
                if pred_word_idx >= len(gt_dlabox_to_predtext_merged[gt_dlabox_idx]):
                    pred_word_str = ''
                else:

                    pred_word_str = gt_dlabox_to_predtext_merged[gt_dlabox_idx][pred_word_idx]
                    if pred_word_str == gt_word_str:
                        n_correct += 1
            else:
                pred_word_str = ''

            cer, lev = evaluate_cer_lev(gt_word_str, pred_word_str)
            # if lev == 0:
                # n_correct += 1

            total_cer += cer
            cer_count += 1
            
            # print([gt_dlabox_idx, gt_word_idx, gt_word_str, pred_word_idx, pred_word_str, cer, lev])
            all_gt_word_pred_word_scores.append([gt_dlabox_idx, gt_word_idx, gt_word_str, pred_word_idx, pred_word_str, cer, lev])


    # Including pred_word without gt_word
    # for gt_dlabox_idx, this_box_all_gt_words_idx in enumerate(pred_gt_mapping_all_box):
    #     # # TODO: punish missing boxes
    #     # if gt_dlabox_to_predbox_merged.get(gt_dla_box_idx, None) is None: 
    #     #     continue
        
    #     for pred_word_idx, gt_word_idx in enumerate(this_box_all_gt_words_idx):
    #         if gt_word_idx == -1:
    #             gt_word_str = ''
    #             # import pdb; pdb.set_trace()
    #             pred_word_str = gt_dlabox_to_predtext_merged[gt_dlabox_idx][pred_word_idx]
    #             cer, lev = evaluate_cer_lev(gt_word_str, pred_word_str)
    #             total_cer += cer
    #             cer_count += 1
    #             all_gt_word_pred_word_scores.append([gt_dlabox_idx, gt_word_idx, gt_word_str, pred_word_idx, pred_word_str, cer, lev])

    if cer_count != 0:
        avg_cer = total_cer / cer_count
    else:
        avg_cer = 0

    # avg_cer = cer_count/
    if cer_count != 0:
        avg_acc = n_correct / cer_count
        # avg_precision = n_correct/pred_words_count
    else:
        avg_acc = 1.0
        avg_precision = 1.0

    avg_precision = (n_correct/pred_words_count) if pred_words_count != 0 else 1.0
    total_den = avg_precision + avg_acc
    avg_f1 = 2*avg_precision*avg_acc/(avg_precision + avg_acc) if total_den > 1e-5 else 0.0


    


    store_in_csv(cer_csv_directory, file_stem, all_gt_word_pred_word_scores, avg_cer, avg_acc, avg_precision, avg_f1, avg_csv_file_path)

    # print('all_predicted_text_region_bbox', all_predicted_text_region_bbox)
    # print(len(all_annotated_words))
    # print('all_annotated_words_bbox', all_annotated_words_bbox)
    # print('len(all_annotated_words_bbox)', len(all_annotated_words_bbox))
    # print('pred_text', len(pred_text))
    # total_num_pred_words = sum(len(sublist) for sublist in pred_text)
    # print(total_num_pred_words)
    # print('roi', roi)
    # print('gt_layout_bbox_list', gt_layout_bbox_list)
    
    return deepcopy_pred_dlabox_to_gt_dlabox_mapper, deepcopy_gt_dlabox_to_pred_superbox, deepcopy_all_gt_text_region_bbox, deepcopy_all_predicted_text_region_bbox
    

def get_centroid(coordinate_list):
    x,y,w,h = coordinate_list
    centroid_x = (x+x+w)/2
    centroid_y = (y+y+h)/2
    return [centroid_x, centroid_y]


import jiwer

def reconstruction_v2_avg_cer_avg_wer_text_level(image, pred_dlabox_to_gt_dlabox_mapper, gt_dlabox_to_pred_superbox, 
                                                all_gt_text_region_bbox, all_predicted_text_region_bbox, roi, gt_layout_text_list, file_stem):

    # print('v2')
    # print('pred_dlabox_to_gt_dlabox_mapper', pred_dlabox_to_gt_dlabox_mapper)
    # print('gt_dlabox_to_pred_superbox', gt_dlabox_to_pred_superbox)
    # print('all_gt_text_region_bbox', all_gt_text_region_bbox)
    # print('all_predicted_text_region_bbox', all_predicted_text_region_bbox)

    # gt_dlabox_to_pred_superbox[2] = [1,2]    
    # print(gt_dlabox_to_pred_superbox)
    # We sort the broken gt dla-box components in the prediction
    # Reverse is not considered !!
    for gt_dlabox_idx, all_pred_dlabox_idx in gt_dlabox_to_pred_superbox.items():
        all_centroids = []
        bad_box = False
        for this_pred_dlabox_idx in all_pred_dlabox_idx:
            if this_pred_dlabox_idx == -1:
                bad_box = True
                continue
            
            bbox = all_predicted_text_region_bbox[this_pred_dlabox_idx]
            this_centroid = [this_pred_dlabox_idx, get_centroid(bbox)]
            all_centroids.append(this_centroid)
        
        # if bad_box: continue
        
        # print('all_centroids', all_centroids)
        sorted_centroid_by_y = sorted(all_centroids, key = lambda x: x[1][1])
        # print('sorted_centroid_by_y', sorted_centroid_by_y)

        sorted_pred_boxes = [idx[0] for idx in sorted_centroid_by_y]
        # print('sorted_pred_boxes', sorted_pred_boxes)
        gt_dlabox_to_pred_superbox[gt_dlabox_idx] = sorted_pred_boxes

    # print('gt_dlabox_to_pred_superbox', gt_dlabox_to_pred_superbox)

    pred_text_region_wise_text = []
    for info_dict in roi:
        pred_text_region_wise_text.append(info_dict['text'])

    # print(text_region_wise_text)
    
    
    total_cer = 0
    total_wer = 0
    box_count = 0
    
    all_gt_dla_box_text = []
    all_pred_dla_box_text = []
    box_seperator = '\n'
    
    
    for gt_dla_box_idx, pred_dla_box_indices in gt_dlabox_to_pred_superbox.items():
        gt_dla_box_text = gt_layout_text_list[gt_dla_box_idx]
        if len(gt_dla_box_text) == 0: 
            continue
            # Due to filtering non-bengali stuffs resulting in empty string in box
        pred_dla_box_text = ''
        seperator = ' '
        # We are penalizing the dla box for which there is not pred box
        # if len(pred_dla_box_indices) == 0: continue # avoid the above penalty
        # for pred_dla_box_idx in pred_dla_box_indices:
        pred_dla_box_text = seperator.join([
            pred_text_region_wise_text[pred_dla_box_idx] for \
                pred_dla_box_idx in pred_dla_box_indices])
        
        all_gt_dla_box_text.append(gt_dla_box_text)
        all_pred_dla_box_text.append(pred_dla_box_text)
        
        # import pdb; pdb.set_trace()
        # TODO: check output of broken dla box by counting num children
        # print(repr(gt_dla_box_text), repr(pred_dla_box_text))
        print("GT: ", gt_dla_box_text)
        print("HP: ", pred_dla_box_text)
        # Paragraph level wer, cer
        # jiwer_output = jiwer.process_words(gt_dla_box_text, pred_dla_box_text)
        # print(jiwer.visualize_alignment(jiwer_output))
        # try:
        #     total_cer += jiwer.cer(gt_dla_box_text, pred_dla_box_text)
        # except:
        #     import pdb; pdb.set_trace()
            
        # total_wer += jiwer.wer(gt_dla_box_text, pred_dla_box_text)
        
        box_count += 1
    
    # avg_cer = (total_cer / box_count) if box_count > 0 else 0.0
    # avg_wer = (total_wer / box_count) if box_count > 0 else 0.0
    # print("Avg of individuals:", avg_cer, avg_wer)
    
    import random
    seed = 5 # 4
    random.Random(seed).shuffle(all_gt_dla_box_text)
    random.Random(seed).shuffle(all_pred_dla_box_text)
    
    all_gt_dla_box_text = box_seperator.join(all_gt_dla_box_text)
    all_pred_dla_box_text = box_seperator.join(all_pred_dla_box_text)
    
    jiwer_output = jiwer.process_words(all_gt_dla_box_text, all_pred_dla_box_text)
    # print(jiwer.visualize_alignment(jiwer_output))
    cer = jiwer.cer(all_gt_dla_box_text, all_pred_dla_box_text)
    wer = jiwer.wer(all_gt_dla_box_text, all_pred_dla_box_text)
    print("Merged all box level:", cer, wer)
    
    reconstruction_v2_metric_df.loc[len(reconstruction_v2_metric_df)] = \
        {
        'file_stem': file_stem, 
        # 'box_cer': avg_cer, 'box_wer': avg_wer, 
        'doc_cer': cer, 'doc_wer': wer
        }
    
    reconstruction_v2_metric_df.to_csv(reconstruction_v2_metric_file_path, index=None)


#
if __name__ == '__main__':
    import traceback

    avg_csv_file_path = os.path.join(base_path, "_avg_cer.csv")
    global reconstruction_v1_metric_file_path, reconstruction_v1_summary_file_path
    reconstruction_v1_metric_file_path = os.path.join(base_path, "_reconstruction_v1_metric.csv")
    reconstruction_v1_summary_file_path = os.path.join(base_path, "_reconstruction_v1_summary.csv")
    
    global reconstruction_v2_metric_file_path
    reconstruction_v2_metric_file_path = os.path.join(base_path, "_reconstruction_v2_metric.csv")
    reconstruction_v2_summary_file_path = os.path.join(base_path, "_reconstruction_v2_summary.csv")
    
    global reconstruction_v1_metric_df, reconstruction_v2_metric_df
    reconstruction_v1_metric_df = pd.DataFrame(columns=['file_stem', 'average_NED', 'average_accuracy', 'average_precision', 'average_f1'])
    reconstruction_v2_metric_df = pd.DataFrame(columns=['file_stem', 'box_cer', 'box_wer', 'doc_cer', 'doc_wer'])
    # reconstruction_v1_summary_df
    
    try:
        with open(avg_csv_file_path, mode='w', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # csv_writer.writerow(['file_stem', 'average_CER'])
            csv_writer.writerow(['file_stem', 'average_NED', 'average_accuracy', 'average_precision', 'average_f1'])

    except Exception as e:
        print(f"An error occurred while creating avg_csv file: {e}")

    with open(os.path.join(base_path, badlad_coco_path)) as f:
        badlad_coco_annotation = json.load(f)
    
    recon_v2_gt_all_info_file = os.path.join(base_path, data_domain_directory, 'reconstruction_v2_gt.pkl')
    
    with open(recon_v2_gt_all_info_file, 'rb') as f:
                recon_v2_gt_annotation_with_text = pickle.load(f)

    all_imgs = list(Path(os.path.join(base_path, image_dir)).glob('*'))
    for image_path in tqdm(all_imgs):
        # try:
        if 1:
            png_name = image_path.name
            file_stem = image_path.stem
            # if file_stem != '0b0e7f70-1225-4ed5-b5e2-9cc5fb9795a6':
                # continue
            
            # if file_stem != "000f02a5-e452-418b-b9a9-316af59d9c11":
            # if file_stem != '00a9f4ae-1425-4976-a2dc-6da331aa3b32':
                # continue
            # if file_stem != 'b9c2b569-575c-4227-8437-e3b026ff669e':
            #     continue
            # if file_stem != 'd3f09723-2b93-4c28-8166-c95daa11de79': # gt box empty??
                # continue
            roi_file = os.path.join(base_path, roi_directory, file_stem+roi_suffix)
            predicted_text_file = os.path.join(base_path, predicted_text_directory, file_stem+pred_text_suffix)
            paragraph_wise_predicted_word_box_file =  os.path.join(base_path, paragraph_wise_predicted_word_box_directory, file_stem+paragraph_wise_predicted_word_box_suffix)
            word_gc_all_box_all_lines_file = os.path.join(base_path, word_gc_all_box_all_lines_directory, file_stem+word_gc_all_box_all_lines_suffix)

            word_annotation_file = os.path.join(base_path, word_annotation_directory, file_stem+'.json')
            
            
            image = cv2.imread(str(image_path))
            
            with open(predicted_text_file, 'rb') as f:
                pred_text = pickle.load(f)
            
            with open(roi_file, 'rb') as f:
                roi = pickle.load(f)

            with open(word_gc_all_box_all_lines_file, 'rb') as f:
                word_gc_all_box_all_lines = pickle.load(f)

            with open(paragraph_wise_predicted_word_box_file, 'rb') as f:
                paragraph_wise_predicted_word_box = pickle.load(f)

            with open(word_annotation_file) as f:
                word_annotation = json.load(f)    


            size = os.path.getsize(image_path)
            json_key = png_name + str(size)
            word_annotation = word_annotation[json_key]

            id = list(filter(lambda image: image['file_name']==png_name, badlad_coco_annotation['images']))[0]['id']
            # print('id', id)

            gt_bbox = list(filter(lambda annotations: annotations['image_id']==id and (annotations['category_id'] in [0,1]), badlad_coco_annotation['annotations']))
            gt_layout_list = list(map(lambda text: [text['bbox'], text['id']], gt_bbox))
            # print(gt_layout_bbox_list)
            gt_layout_bbox_list = []
            gt_layout_text_list = []
            recon_v2_gt_annotation_this_image = recon_v2_gt_annotation_with_text[png_name]
            
            for gt_box in gt_layout_list:
                gt_box_text = ''
                gt_box_bbox = None
                for text_gt_box in recon_v2_gt_annotation_this_image:
                    if gt_box[1] == text_gt_box['id']:
                        gt_box_text = text_gt_box['text']
                        gt_box_bbox = gt_box[0]
                        break
                # print(gt_box_text)
                assert gt_box_bbox is not None, "COCO id mismatch"
                gt_layout_bbox_list.append(gt_box_bbox)
                gt_layout_text_list.append(gt_box_text)
            
            # print(gt_layout_bbox_list)
            # print(gt_layout_text_list)
            # exit()
                    

            # print('roi_file:', roi_file)
            # print('image_path:', image_path)
            # print('png_name:', png_name)
            # print('file_stem:', file_stem)
            print(f'Evaluating {png_name}')
            
            
            
            # import pdb; pdb.set_trace()
        
        
            pred_dlabox_to_gt_dlabox_mapper, gt_dlabox_to_pred_superbox, all_gt_text_region_bbox, all_predicted_text_region_bbox = reconstruction_v1_avg_cer(image, pred_text, roi, word_annotation, gt_layout_bbox_list, # paragraph_wise_predicted_word_box, 
                                    word_gc_all_box_all_lines, 
                                    file_stem, cer_csv_directory, avg_csv_file_path)


            
            
            # print(recon_v2_gt_annotation_with_text)
            # import pdb; pdb.set_trace()
            
            
            reconstruction_v2_avg_cer_avg_wer_text_level(image, pred_dlabox_to_gt_dlabox_mapper, gt_dlabox_to_pred_superbox, all_gt_text_region_bbox, all_predicted_text_region_bbox, roi, gt_layout_text_list, file_stem)

            
            print(f'Successful {png_name}!')
        # except Exception as e:
            # print(f"Error {e}")
            # print(traceback.format_exc())
    
    reconstruction_v1_metric_df.sort_values('average_NED', inplace=True)
    reconstruction_v1_metric_df.to_csv(reconstruction_v1_metric_file_path, index=None)
    reconstruction_v2_metric_df.to_csv(reconstruction_v2_metric_file_path, index=None)
    
    # reconstruction_v1_summary_file_path
    reconstruction_v1_summary_df = pd.DataFrame({
        'mean_average_NED': [reconstruction_v1_metric_df[['average_NED']].mean()],
        'std_average_NED': [reconstruction_v1_metric_df[['average_NED']].std()],
        'mean_average_accuracy': [reconstruction_v1_metric_df[['average_accuracy']].mean()],
        'std_average_accuracy': [reconstruction_v1_metric_df[['average_accuracy']].std()],
    })
    
    reconstruction_v1_summary_df.to_csv(reconstruction_v1_summary_file_path, index=None)

'''
python -m pdb -c continue reconstruction_v1v2.py
'''