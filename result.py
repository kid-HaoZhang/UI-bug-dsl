import json
import os
from config import Config

class Result:
    coco = dict()
    categories = []
    annotations = []
    images = []
    catenames = []

    def init(self):
        if os.path.exists(Config.output_json):
            self.coco = json.load(Config.output_json)
            self.categories = self.coco['categories']
            self.annotations = self.coco['annotations']
            self.images = self.coco['images']
            for cat in self.categories:
                self.catenames.append(cat['name'])
        else:
            self.coco['categories'] = []
            self.coco['annotations'] = []
            self.coco['images'] = []

    def coco2file(self):
        pass
    
    def add_cat(self, name: str):
        if self.catenames.count(name) != 0:
            return
        self.catenames.append(name)
        self.categories.append({'id': len(self.catenames), 'name': name, 'supercategory': 'app'})
        