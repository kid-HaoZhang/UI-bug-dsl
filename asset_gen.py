import os
import re
import cv2

def all_type():
    all = set()
    p = r'D:\HongMeng\archive\rico_dataset_v0.1_semantic_annotations\semantic_annotations'
    for file in os.listdir(p):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(p, file), 'r', encoding='UTF-8') as c:
            for line in c:
                line = line.strip()
                if line.startswith('\"componentLabel\"'):
                    type = line
                    all.add(re.sub('\"', '', type))
    print(all)
    '''
    'Date', 'Drawer', 'Toolbar', 'Slider', 'Bottom', 'Card', 'Background', 
    'Icon', 'Radio', 'Button', 'Number', 'Multi-Tab', 'On/Off', 'Checkbox', 
    'Input', 'Modal', 'Video', 'Advertisement', 'Map', 'Web', 'Pager', 'Text', 
    'Image', 'List'
    '''

if __name__ == '__main__':
    # all_type()
    p = r'D:\HongMeng\archive\unique_uis\combined\100.jpg'
    img = cv2.imread(p)
    background = img[0,0]
    h, w = img.shape[:2]
    # 创建一个和图片大小相同的掩码，左半部分是全黑，右半部分是[0,0]处的像素值
    img[:,w//2:] = background
    cv2.imshow(' ', img)
    cv2.waitKey(0)
    # i = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    # width = 1440
    # height = 2560
    # screen_width = 540
    # screen_height = 960
    # y_ratio = screen_width/width
    # x_ratio = screen_height/height
    # y1, y2, x1, x2 = int(144*y_ratio), int(220*y_ratio), int(196*x_ratio), int(1272*x_ratio)
    # tmp = i[y1:y2, x1:x2]
    # i[y1+20:y2+20, x1+20:x2+20] = tmp
    # cv2.imshow(' ', i)
    # cv2.waitKey(0)