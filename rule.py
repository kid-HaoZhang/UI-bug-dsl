widget_type=['Web View', 'Date Picker', 'Text Button', 'On/Off Switch', 
             'Toolbar', 'Background Image', 'Card', 'Number Stepper', 'Drawer',
               'Image', 'Modal', 'Pager Indicator', 'Checkbox', 
               'List Item', 'Advertisement', 
               'Bottom Navigation', 'Slider']

class Tran:
    w_type = None
    position = ""
    func = None
    copy = False

    def __init__(self):
        pass


class rule:
    type_name = ""
    trans = []

    def __init__(self, name: str):
        self.type_name = name

    def add_trans(self, tran: Tran):
        self.trans.append(tran)

    