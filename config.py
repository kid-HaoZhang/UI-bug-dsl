import json
import os

class Config():
    gennum = 10
    UI_dir = ""
    annotation_dir = ""
    annotation_file = ""
    proj_dir = ""
    result_dir = ""
    result_UI_dir = ""
    dataset_style = ""
    output_json = ""

    def parse_config(self):
        configs = json.loads(open('config.josn', 'r').read())
        self.gennum = configs["gennum"]
        self.UI_dir = configs["UI_dir"]
        self.annotation_dir = configs["annotation_dir"]
        self.annotation_file = configs["annotation_file"]
        self.proj_dir = os.path.abspath('.')
        self.result_dir = os.path.join(self.proj_dir, 'result')
        self.result_UI_dir = os.path.join(self.result_dir, "bugUI")
        self.dataset_style = configs['dataset_style']
        self.output_json = configs['output_json']