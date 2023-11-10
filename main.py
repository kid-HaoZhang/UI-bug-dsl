import os
import re
import shutil
import sys
import importlib
from rule import *
from gen_bug import gen_bug
from config import Config
from cocoHelper import cocoHelper
from result import Result

dsl_path = 'bug.dsl'

def gen_rules():
    Rules = []
    with open('bug.dsl', 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            bug_name, line = line.split(':') # bug_name, rule
            idx = line.index(' ')
            w_type = line[:idx]
            line = line[idx + 1:] # widget_name, keep/background rule|...
            idx = line.index(' ')
            keep = line[:idx] == 'keep'
            line = line[idx + 1:]

            w_type = w_type.capitalize()
            e = False
            for w in widget_types:
                if w_type == w.capitalize():
                    e = True
                    break
            if not e:
                continue
            Rule = rule(bug_name, w_type, keep)

            trans = line.split('|')
            for r in trans:
                tran = Tran()
                parts = r.split(' ')
                if len(parts) < 3:
                    print('rule wrong format:', line)
                tran.focus = parts[0] == 'focus'
                tran.func = parts[1]
                tran.position = parts[2]
                Rule.add_trans(tran)
            Rules.append(Rule)
    return Rules

def main():
    Config.parse_config()
    cocoHelper.init()
    Result.init()
    for rule in gen_rules():
        gen_bug(rule)

if __name__ == "__main__":
    main()
