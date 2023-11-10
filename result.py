import json
import os

import cv2
from config import Config
from modules.cv_helper import save_img

class Result:
    coco = dict()
    categories = []
    annotations = []
    images = []
    catdict = dict()
    @staticmethod
    def init():
        if os.path.exists(Config.result_json):
            Result.coco = json.load(Config.result_json)
            Result.categories = Result.coco['categories']
            Result.annotations = Result.coco['annotations']
            Result.images = Result.coco['images']
            for cat in Result.categories:
                Result.catdict[cat['name']] = cat
        else:
            Result.categories = []
            Result.annotations = []
            Result.images = []
    @staticmethod
    def coco2file():
        Result.coco['categories'] = Result.categories
        Result.coco['annotations'] = Result.annotations
        Result.coco['images'] = Result.images
        with open(Config.result_json, 'w') as f:
            json.dump(Result.coco, f)
    @staticmethod
    def add_cat(name: str):
        if name in Result.catdict.keys():
            return
        cat = {'id': len(Result.catdict.keys()), 'name': name, 'supercategory': 'app'}
        Result.categories.append(cat)
        Result.catdict[name] = cat
    @staticmethod
    def get_cat_id(cat_name):
        Result.add_cat(cat_name)
        return Result.catdict[cat_name]['id']
    @staticmethod
    def add_annotations(img: cv2.Mat, bug_img_name: str, width, height, cat_name: str, img_name:str, bboxs: list):
        '''example
        {
            "file_name": "Wireframes__wf_98.jpg",
            "id": 480,
            "width": 375,
            "height": 812
        }
        bbox can be [x1, y1, x2, y2]
        {
            "area": 3808.0,
            "bbox": [
                135.0,
                301.0,
                136.0,
                28.0
            ],
            "category_id": 1,
            "id": 0,
            "image_id": 0,
            "iscrowd": 0,
            "segmentation": [
                [
                    135.0,
                    301.0,
                    271.0,
                    301.0,
                    271.0,
                    329.0,
                    135.0,
                    329.0
                ]
            ]
        }
        '''
        if len(bboxs) == 0:
            return
        img_pth = os.path.join(Config.result_UI_dir, bug_img_name)
        if os.path.exists(img_pth):
            for i in range(50):
                bug_img_name =  cat_name + '_' + Config.version_name + str(i) + img_name
                img_pth = os.path.join(Config.result_UI_dir, bug_img_name)
                if not os.path.exists(img_pth):
                    break
        save_img(img_pth, img)
        # 添加新img
        img_info = {'file_name':bug_img_name, 'id': len(Result.images), 'width': width, 'height': height}
        Result.images.append(img_info)
        # 添加新annotations
        cat_id = Result.get_cat_id(cat_name)
        img_id = img_info['id']
        segmentation = [[]]
        area = height*width
        for bbox in bboxs:
            id = len(Result.annotations)
            annotation = {
                "area": area,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "category_id": cat_id,
                "id": id,
                "image_id": img_id,
                "iscrowd": 0,
                "segmentation": segmentation
            }
            Result.annotations.append(annotation)

