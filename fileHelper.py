import os
import shutil
from config import Config

class FileHelper:
    @staticmethod
    def new_dir(dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)    
        os.mkdir(dir)
    
    @staticmethod
    def get_image_json_pairs() -> []:
        '''
        返回image json图片对
        '''
        if Config.dataset_style == "rico":
            return FileHelper.get_pairs_rico()
        elif Config.dataset_style == "coco":
            return FileHelper.get_pairs_vins()
        else:
            print("no dataset")
            return []
    
    @staticmethod
    def get_pairs_rico() -> []:
        pairs = []
        for jpg_file in os.listdir(Config.UI_dir):
            if not jpg_file.endswith('.jpg'):
                continue
            name = jpg_file.split('.')[0]
            annotation_file = os.path.join(Config.annotation_dir, name + '.json')
            jpg_path = os.path.join(Config.UI_dir, jpg_file)
            pairs.append([jpg_path, annotation_file])

    @staticmethod
    def get_pairs_vins() -> []:
        pass

    @staticmethod
    def join(path, *paths):
        return os.path.join(path, paths)