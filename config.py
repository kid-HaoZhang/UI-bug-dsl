import json
import os
import shutil

def new_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)    
    os.mkdir(dir)

class Config():
    gennum = 10
    UI_dir = ""
    annotation_dir = ""
    annotation_file = ""
    proj_dir = ""
    result_dir = ""
    result_UI_dir = ""
    result_json = ""
    dataset_style = ""
    version_name = "ori"
    @staticmethod
    def parse_config():
        Config.proj_dir = os.path.abspath('.')
        configs = json.loads(open(os.path.join(Config.proj_dir, 'config.json'), 'r').read())
        Config.gennum = configs["gennum"]
        Config.UI_dir = configs["UI_dir"]
        Config.annotation_dir = configs["annotation_dir"]
        Config.annotation_file = configs["annotation_file"]
        Config.result_dir = os.path.join(Config.proj_dir, 'result')
        # new_dir(Config.result_dir)
        Config.result_UI_dir = os.path.join(Config.result_dir, "bugUI")
        # new_dir(Config.result_UI_dir)
        Config.result_json = os.path.join(Config.result_dir, 'annotations.json')
        Config.dataset_style = configs['dataset_style']
        Config.version_name = configs['version_name']
        
if __name__ == "__main__":
    Config.parse_config()