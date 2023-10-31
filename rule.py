widget_types=['Web View', 'Date Picker', 'Text Button', 'On/Off Switch', 
             'Toolbar', 'Background Image', 'Card', 'Number Stepper', 'Drawer',
               'Image', 'Modal', 'Pager Indicator', 'Checkbox', 
               'List Item', 'Advertisement', 
               'Bottom Navigation', 'Slider']

class Tran:
    position = ""
    copy = False
    func = None

    def __init__(self):
        pass


class rule:
    bug_name = ""
    trans = []
    widget_type = ""

    def __init__(self, bug_name: str, w_name: str):
        self.bug_name = bug_name
        self.widget_type = w_name

    def add_trans(self, tran: Tran):
        self.trans.append(tran)

    