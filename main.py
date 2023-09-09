import re
import sys
import importlib
from rule import *
from gen_bug import gen_bug

with open('bug.dsl', 'r') as file:
    for line in file:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        bug_name, line = line.split(':')

        Rule = rule(bug_name)

        rules = line.split('|')
        for r in rules:
            tran = Tran()

            parts = r.split(' ')
            w_type = parts[0][0].capitalize() + parts[0][1:]
            w_type = re.sub('_', ' ', w_type)
            if w_type not in widget_type:
                print('not exist widget_type:', line)
                continue
            tran.w_type = w_type
            
            tran.position = parts[1]
            if len(parts) > 2:
                if parts[2]=="copy":
                    tran.copy = True
                else:
                    tran.copy = False
                    tran.func = parts[2]
            Rule.add_trans(tran)
        
        gen_bug(Rule)

