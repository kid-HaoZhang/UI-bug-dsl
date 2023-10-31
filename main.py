import os
import re
import shutil
import sys
import importlib
from rule import *
from gen_bug import gen_bug
from config import Config
from cocoHelper import cocoHelper

dsl_path = 'bug.dsl'

def gen_rules():
    Rules = []
    with open('bug.dsl', 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            bug_name, line = line.split(':') # bug_name, rule
            w_type, line = line.split(' ') # widget_name, rule|...
            w_type = w_type[0].capitalize() + w_type[1:]
            if w_type not in widget_types:
                print('not exist widget_type:', line)
                continue
            Rule = rule(bug_name, w_type)

            trans = line.split('|')
            for r in trans:
                tran = Tran()
                parts = r.split(' ')
                if len(parts) < 3:
                    print('rule wrong format:', line)
                tran.position = parts[0]
                tran.copy = parts[1] == "keep"
                tran.func = parts[2]
                Rule.add_trans(tran)
            Rules.append(Rule)
    return Rules

def main():
    Config.parse_config()
    cocoHelper.init()
    for rule in gen_rules():
        gen_bug(rule)
