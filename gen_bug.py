from collections import defaultdict
import os
from rule import *
from modules.func import *
import cv2
import json
import re
import shutil
from modules.cv_helper import *
from FileHelper import FileHelper
from config import Config
from util import *
from pycocotools.coco import COCO
from cocoHelper import cocoHelper
from result import Result

def show_img(img):
    cv2.imshow('Binary Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exist_type(json_path: str, w_type: str):
    with open(json_path, 'r', encoding='UTF-8') as f:
        return w_type in f.read()

def able_deal(json_path, type_name, able_bounds: list):
    if Config.dataset_style == "rico":
        json_dict = json.load(json_path)
        able_deal_rico(json_dict, type_name, able_bounds)
    elif Config.dataset_style == "coco":
        able_deal_coco()
    
def able_deal_coco():
    pass

def able_deal_rico(node: dict, type_name: str, able_bounds: list):
    if node.get('componentLabel') and (type_name == node['componentLabel'] or type_name == 'widget'):
        able_bounds.append(node['bounds'])
        return 
    if not node.get('children'):
        return
    for c in node.get('children'):
        able_deal(c, type_name, able_bounds)
    
def new_pos(pos: str):  # [a,b,c,d]
    parts = pos[1:-1].split(',')
    return [eval(parts[0]), eval(parts[1]), eval(parts[2]), eval(parts[3])]

def new_pos_str(bounds: list, pos_str: str):
    name = ['X1', 'Y1', 'X2', 'Y2']
    for i in range(4):
        pos_str = re.sub(name[i], str(bounds[i]), pos_str)
    return pos_str

def cover_widget_with_background(img: cv2.Mat, background):
    img[:] = background
    return img

def gen_bug(bug_rule: rule):
    file_name = bug_rule.bug_name
    store_path = Config.result_dir
    FileHelper.new_dir(store_path)

    if Config.dataset_style == "rico":
        for pair in FileHelper.get_image_json_pairs():
            if len(pair) != 2:
                print("wrong pair")
                return
            gen_bug_one_UI(bug_rule, pair[0], pair[1])
    elif Config.dataset_style == "coco":
        gen_bug_coco(bug_rule)

def gen_bug_coco(bug_rule: rule):
    img2bboxs = defaultdict(list)
    for anno_id in cocoHelper.anno_ids:
        anno = cocoHelper.anno_dict[anno_id]
        cat = cocoHelper.cat_dict[cocoHelper.anno['category_id']]
        if cat['name'].capitalize() != bug_rule.widget_type.capitalize():
            continue
        bbox = anno['bbox']
        bound = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] # x1, y1, x2, y2
        img2bboxs[anno['image_id']].append(bound)
    for img_id, bounds in img2bboxs:
        img_info = cocoHelper.img_dict[img_id]
        img = cv2.imread(FileHelper.join(Config.UI_dir, img_info['file_name']))
        img = cv2.resize(img, (img_info['width'], img_info['height']))
        gen_bug_UI_bounds(bug_rule, img, bounds)
        
def gen_bug_UI_bounds(bug_rule: rule, img: cv2.Mat, bounds: [[]]):
    bugname = bug_rule.bug_name
    Result.add_cat(bugname) # new category
    
    select_index = select_random_numbers(len(bounds))
    for idx in select_index:
        bound = bounds[idx]
        x1, y1, x2, y2 = bound[0], bound[1], bound[2], bound[3]
        widget_img = remove_background(img[y1:y2, x1:x2]) # 认为已经是精确bound, 且去除掉背景
        for tran in bug_rule.trans:
            widget_new_pos = new_pos(new_pos_str(bound, tran.position))
            widget_new_pos = [max(0, widget_new_pos[0]), max(0, widget_new_pos[1]), 
                            min(widget_new_pos[2], width), min(widget_new_pos[3], height)]
            new_widget_width, new_widget_height = widget_new_pos[2] - widget_new_pos[0], widget_new_pos[3] - widget_new_pos[1]
            
            if tran.copy:
                img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = widget_img[:new_widget_height, :new_widget_width]
            else:
                img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = background
            
            new_widget = widget_img.copy()
            if tran.func is not None:
                new_widget = eval(tran.func)(new_widget, background)
            overlap()
            img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = new_widget[:new_widget_height, :new_widget_width]


def gen_bug_one_UI(bug_rule: rule, UI_path, annotation_file):
    if not exist_type(annotation_file, bug_rule.widget_type): # 不存在要处理的type
        return

def deal_trans_v1(bug_rule: rule, UI_path, annotation_file):
    img = cv2.imread(UI_path, cv2.IMREAD_COLOR)
    width, height = 1440, 2560
    img = cv2.resize(img, (width, height))

    json_dict = json.loads(open(annotation_file, 'r', encoding='UTF-8').read())

    able_bounds=[]
    able_deal(json_dict, bug_rule.widget_type, able_bounds) # 所有符合type的组件bound

    select_index = select_random_numbers(len(able_bounds))
    for idx in select_index:
        bound = able_bounds[idx]
        y1, y2, x1, x2 = bound[1], bound[3], bound[0], bound[2]
        bound = [x1,y1,x2,y2]
        widget_img = img[y1:y2, x1:x2]
        # 找到组件更精确的bound
        background = find_background_color(widget_img)
        if (t := find_icon(widget_img)) is not None:
            bound = [x1+t[0], y1+t[1], x1+t[0]+t[2], y1+t[1]+t[3]]
        widget_img = img[bound[1]:bound[3], bound[0]:bound[2]]
        
        for tran in bug_rule.trans:
            widget_new_pos = new_pos(new_pos_str(bound, tran.position))
            widget_new_pos = [max(0, widget_new_pos[0]), max(0, widget_new_pos[1]), 
                            min(widget_new_pos[2], width), min(widget_new_pos[3], height)]
            new_widget_width, new_widget_height = widget_new_pos[2] - widget_new_pos[0], widget_new_pos[3] - widget_new_pos[1]

            if tran.copy:
                img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = widget_img[:new_widget_height, :new_widget_width]
            else:
                img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = background

            new_widget = widget_img.copy()
            if tran.func is not None:
                new_widget = eval(tran.func)(new_widget, background)
            img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = new_widget[:new_widget_height, :new_widget_width]

        return img
    


@DeprecationWarning
def deal_with_tran(img: cv2.Mat, json_dict: dict, tran: Tran):  
    '''
    
    处理一条tran
    '''
    width, height = 1440, 2560
    screen_height, screen_width, _c = img.shape
    x_ratio, y_ratio = screen_width / width, screen_height / height
    
    able_bounds=[]
    able_deal(json_dict, tran.w_type, able_bounds) # 所有符合type的组件bound

    if len(able_bounds) == 0:
        return None

    for bounds in able_bounds:  # x1, y1, x2, y2
        y1, y2, x1, x2 = int(bounds[1]*y_ratio), int(bounds[3]*y_ratio), int(bounds[0]*x_ratio), int(bounds[2]*x_ratio)
        bounds = [x1,y1,x2,y2]
        widget_img = img[y1:y2, x1:x2]
        background = find_background_color(widget_img)
        if (t := find_icon(widget_img)) is not None:
            bounds = [x1+t[0], y1+t[1], x1+t[0]+t[2], y1+t[1]+t[3]]
        widget_img = img[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        widget_new_pos = new_pos(new_pos_str(bounds, tran.position))
        widget_new_pos = [max(0, widget_new_pos[0]), max(0, widget_new_pos[1]), 
                        min(widget_new_pos[2], screen_width), min(widget_new_pos[3], screen_height)]
        new_widget_width, new_widget_height = widget_new_pos[2] - widget_new_pos[0], widget_new_pos[3] - widget_new_pos[1]

        if tran.copy:
            img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = widget_img[:new_widget_height, :new_widget_width]
            return img

        if widget_new_pos != bounds:
            widget_img = widget_img.copy()
            img[bounds[1]:bounds[3], bounds[0]:bounds[2]] = background  # 原来位置变为背景

            if tran.func is not None:
                new_widget = eval(tran.func)(widget_img, background)
                img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = new_widget[:new_widget_height, :new_widget_width]
            else:
                img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = widget_img[:new_widget_height, :new_widget_width]
            # TODO: 记录生成的坐标和bug类型
        else:
            widget_img = widget_img.copy()
            img[bounds[1]:bounds[3], bounds[0]:bounds[2]] = background  # 原来位置变为背景
            if tran.func is not None:
                new_widget = eval(tran.func)(widget_img, background)
                img[bounds[1]:bounds[3], bounds[0]:bounds[2]] = new_widget[:new_widget_height, :new_widget_width]
            # TODO: 记录生成的坐标和bug类型
    return img
