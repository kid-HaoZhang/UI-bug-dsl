from collections import defaultdict
from pycocotools.coco import COCO
from config import Config

class cocoHelper():
    anno_ids: list[int]
    anno_dict: dict
    cat_ids: list[int]
    cat_dict: dict
    img_ids: list[int]
    img_dict: dict
    @staticmethod
    def init():
        if Config.dataset_style != "coco":
            return
        coco = COCO(annotation_file=Config.annotation_file)
        cocoHelper.anno_ids = coco.getAnnIds()
        cocoHelper.anno_dict = {anno['id']:anno for anno in coco.loadAnns(cocoHelper.anno_ids)}
        cocoHelper.cat_ids = coco.getCatIds()
        cocoHelper.cat_dict = {cat['id']:cat for cat in coco.loadCats(cocoHelper.cat_ids)}
        cocoHelper.img_ids = coco.getImgIds()
        cocoHelper.img_dict = {img['id']:img for img in coco.loadImgs(cocoHelper.img_ids)}
