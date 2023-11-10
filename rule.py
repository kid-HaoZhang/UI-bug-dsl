widget_types=['Web View', 'Date Picker', 'Text Button', 'On/Off Switch', 
             'Toolbar', 'Background Image', 'Card', 'Number Stepper', 'Drawer',
               'Image', 'Modal', 'Pager Indicator', 'Checkbox', 
               'List Item', 'Advertisement', 
               'Bottom Navigation', 'Slider', 'EditText', 'Icon', 'Map', 'Multi_Tab','PageIndicator'
               'Remember', 'Spinner', 'Switch', 'Text', 'TextButton', 'UpperTaskBar','CheckedTextView',
               'Bottom_Navidation', 'BackgroundImage']

class Tran:
    
    def __init__(self):
        self.position = ""
        self.focus = False
        self.func = None
        pass


class rule:

    def __init__(self, bug_name: str, w_name: str, keep: bool):
        self.bug_name = bug_name
        self.widget_type = w_name
        self.keep = keep
        self.trans = []

    def add_trans(self, tran: Tran):
        self.trans.append(tran)

    