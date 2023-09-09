import os
from rule import *
from modules.func import *
import cv2
import json
import re
import shutil
from modules.bound import find_background_color

gen_number = 1

proj_dir = os.path.abspath('.')
annotation_dir = f'D:\HongMeng\\archive\\rico_dataset_v0.1_semantic_annotations\semantic_annotations'
UI_dir = r"D:\\HongMeng\\archive\\unique_uis\\combined"
asset_dir = os.path.join(proj_dir, 'Assets')
result_dir = os.path.join(proj_dir, 'result')

def show_img(img):
    cv2.imshow('Binary Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exist_type(json_path: str, w_type: str):
    with open(json_path, 'r', encoding='UTF-8') as f:
        return w_type in f.read()

def able_deal(node: dict, type_name: str, able_bounds: list):
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

def deal_with_tran(img: cv2.Mat, json_dict: dict, tran: Tran):  # 处理一条tran
    width, height = 1440, 2560
    screen_height, screen_width, _c = img.shape
    x_ratio, y_ratio = screen_width / width, screen_height / height
    
    able_bounds=[]
    able_deal(json_dict, tran.w_type, able_bounds)

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



def gen_bug(bug_rule: rule):
    file_name = bug_rule.type_name
    store_path = os.path.join(proj_dir, 'result', file_name)

    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    number = 0

    for jpg_file in os.listdir(UI_dir):
        if not jpg_file.endswith('.jpg'):
            continue
        name = jpg_file.split('.')[0]
        annotation_file = os.path.join(annotation_dir, name + '.json')

        if not exist_type(annotation_file, bug_rule.trans[0].w_type): # 不存在要处理的type
            continue
        
        jpg_path = os.path.join(UI_dir, jpg_file)
        img = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        for tran in bug_rule.trans:
            img = deal_with_tran(img, json.loads(open(annotation_file, 'r', encoding='UTF-8').read()), tran)
            if img is None:
                break
        if img is None:
            continue
        cv2.imwrite(os.path.join(store_path, jpg_file), img)
        number += 1
        if number > gen_number:
            break
