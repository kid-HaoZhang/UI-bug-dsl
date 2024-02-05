from collections import defaultdict
import os
from rule import *
from modules.func import *
import cv2
import json
import re
import shutil
from modules.cv_helper import *
from fileHelper import FileHelper
from config import Config
from util import *
from pycocotools.coco import COCO
from cocoHelper import cocoHelper
from result import Result

def show_img(img):
    return
    print(img.shape)
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
    store_path = Config.result_dir
    # FileHelper.new_dir(store_path)

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
    for anno_id in cocoHelper.anno_ids: # 将可处理的img和box配对
        anno = cocoHelper.anno_dict[anno_id]
        cat = cocoHelper.cat_dict[anno['category_id']]
        if cat['name'].capitalize() != bug_rule.widget_type.capitalize():
            continue
        bbox = anno['bbox']
        bound = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] # x1, y1, x2, y2
        img2bboxs[anno['image_id']].append(bound)
    cnt = 0 
    for img_id, bounds in img2bboxs.items():
        cnt += 1
        if cnt > Config.gennum:
            break
        img_info = cocoHelper.img_dict[img_id]
        img = cv2.imread(os.path.join(Config.UI_dir, img_info['file_name']))
        img = cv2.resize(img, (img_info['width'], img_info['height']))
        gen_bug_UI_bounds(bug_rule, img, bounds, img_info['file_name'])
    Result.coco2file()
        
def expand_widget(bounds: [], width, height):
    re = bounds.copy()
    re[0] = int(0 if re[0] - 10 < 0 else re[0] - 10)
    re[1] = int(0 if re[1] - 10 < 0 else re[1] - 10)
    re[2] = int(re[2] + 10 if re[2] + 10 < width else width)
    re[3] = int(re[3] + 10 if re[3] + 10 < height else height)
    return re

def bug_bound(bound1, bound2:list): # 都是x1, y1, x2, y2的顺序
    if bound1 is None:
        return bound2.copy()
    b = [0 for i in range(4)]
    b[0] = min(bound1[0], bound2[0])
    b[1] = min(bound1[1], bound2[1])
    b[2] = max(bound1[2], bound2[2])
    b[3] = max(bound1[3], bound2[3])
    return b

def gen_bug_UI_bounds(bug_rule: rule, img: cv2.Mat, bounds: [[]], img_name:str):
    bugname = bug_rule.bug_name
    Result.add_cat(bugname) # new category
    height, width, _ = img.shape

    select_index = select_random_numbers(len(bounds))
    bugs_pos = []

    ori_img = img.copy()

    for idx in select_index:
        bound = bounds[idx]
        x1, y1, x2, y2 = map(int, bound)
        expand_bound = expand_widget(bound, width, height)
        background = find_background_color(img[expand_bound[1]:expand_bound[3], expand_bound[0]:expand_bound[2]])
        
        ori_bound_widget = img[y1:y2, x1:x2].copy()
        exp_bound_widget = img[expand_bound[1]:expand_bound[3], expand_bound[0]:expand_bound[2]].copy()

        if not bug_rule.keep:
            img[y1:y2, x1:x2] = background

        widget_img = ori_bound_widget
        use_bound = bound
        show_img(widget_img)
        gened = True
        bug_pos = None
        for tran in bug_rule.trans:
            if tran.focus:
                if bug_rule.widget_type == 'Text':
                    text = get_text(widget_img)
                    if tran.func == 'null':
                        text = 'null'
                    if text == "":
                        gened = False
                        continue
                    # TODO：暂时认为对于focus text只会做偏移/null等，不会做复杂的改变
                    txt_color = get_txt_color(widget_img)
                    txt_size = get_txt_size(widget_img)
                    ori_new_pos = new_pos(new_pos_str(bound, tran.position))
                    ori_new_pos = [int(a) for a in ori_new_pos]
                    ori_new_pos = [max(0, ori_new_pos[0]), max(0, ori_new_pos[1]), 
                            min(ori_new_pos[2], width), min(ori_new_pos[3], height)]
                    new_widget_width, new_widget_height = ori_new_pos[2] - ori_new_pos[0], ori_new_pos[3] - ori_new_pos[1]
                    bug_pos = bug_bound(bug_pos, ori_new_pos)
                    put_text(img, ori_new_pos[0], ori_new_pos[1], text, new_widget_height, txt_size, txt_color)
                    
                    if get_text(img[ori_new_pos[1]:ori_new_pos[3], ori_new_pos[0]:ori_new_pos[2]]) == '':
                        gened = False
                    
                    continue
                else:
                    widget_img = remove_bg(exp_bound_widget) #  且去除掉背景, 会返回一个四通道的图片
                    show_img(widget_img)
                    use_bound = expand_bound
            
            # use_widget = ori_img[use_bound[1]:use_bound[3], use_bound[0]:use_bound[2]].copy()
            widget_new_pos = new_pos(new_pos_str(use_bound, tran.position))
            widget_new_pos = [int(a) for a in widget_new_pos]
            widget_new_pos = [max(0, widget_new_pos[0]), max(0, widget_new_pos[1]), 
                            min(widget_new_pos[2], width), min(widget_new_pos[3], height)]
            
            ori_new_pos = new_pos(new_pos_str(bound, tran.position))
            ori_new_pos = [max(0, ori_new_pos[0]), max(0, ori_new_pos[1]), 
                            min(ori_new_pos[2], width), min(ori_new_pos[3], height)]

            new_widget_width, new_widget_height = widget_new_pos[2] - widget_new_pos[0], widget_new_pos[3] - widget_new_pos[1]
            
            widget_img = cv2.resize(widget_img, dsize=(new_widget_width, new_widget_height), interpolation=cv2.INTER_AREA)
            if tran.func is not None:
                new_widget = eval(tran.func)(widget_img, background)
                show_img(new_widget)
            
            img = overlay(img, widget_new_pos[0], widget_new_pos[1], new_widget, new_widget_width, new_widget_height)
            show_img(img)
            bug_pos = bug_bound(bug_pos, ori_new_pos) # 还是以最精确的bound来
        
        if gened:
            bugs_pos.append(bug_pos)
    # img contains a few bugs now in bugs_pos
    bug_img_name =  bugname + '_' + Config.version_name + img_name
    Result.add_annotations(img, bug_img_name, width, height, bugname, img_name, bugs_pos)
            # img[widget_new_pos[1]:widget_new_pos[3], widget_new_pos[0]:widget_new_pos[2]] = new_widget[:new_widget_height, :new_widget_width]

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
