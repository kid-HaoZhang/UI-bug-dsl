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

    def init(self):
        if Config.dataset_style != "coco":
            return
        coco = COCO(annotation_file=Config.annotation_file)
        self.anno_ids = coco.getAnnIds()
        self.anno_dict = {anno['id']:anno for anno in coco.loadAnns(self.anno_ids)}
        self.cat_ids = coco.getCatIds()
        self.cat_dict = {cat['id']:cat for cat in coco.loadCats(self.cat_ids)}
        self.img_ids = coco.getImgIds()
        self.img_dict = {img['id']:img for img in coco.loadImgs(self.img_ids)}
