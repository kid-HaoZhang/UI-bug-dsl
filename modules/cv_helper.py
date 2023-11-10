from collections import defaultdict
import random
import matplotlib
from rembg import remove, new_session
import cv2
import numpy as np
from PIL import Image
from pymatting import cutout
import pytesseract
from sklearn.cluster import KMeans

def remove_bg(img: cv2.Mat): # 效果不错哦
    model_name = "isnet-general-use"
    session = new_session(model_name)
    output = remove(img, session=session, alpha_matting=True, 
                    alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, 
                    alpha_matting_erode_size=0, post_process_mask=True)
    return output

def get_txt_color(img: cv2.Mat):
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=2)
    labels, centers = clt.fit_predict(img), clt.cluster_centers_
    counts = np.bincount(labels)
    max_index = np.argmax(counts)
    text_color = np.delete(centers, max_index, axis=0)
    hsv = matplotlib.colors.rgb_to_hsv([a / 255.0 for a in text_color[0]])
    hsv[1] = np.clip(hsv[1] + 0.05, 0, 1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv) * 255
    # print(text_color)
    return rgb
    # image_flat = np.reshape(image, (-1, 3))
    # # 统计每个颜色值的出现次数
    # unique_colors, counts = np.unique(image_flat, axis=0, return_counts=True)
    # sorted_indices = np.argsort(-counts)
    # if len(sorted_indices) < 10:
    #     return (0, 0, 0)
    # second_color_index = sorted_indices[10]
    # return unique_colors[second_color_index]

def get_txt_size(img: cv2.Mat):
    text = pytesseract.image_to_boxes(img, lang='eng')
    # 按行分割字符串
    lines = text.split('\n')
    # 遍历每一行
    color_dict = defaultdict(int)
    max_height = 0
    for line in lines:
        # 按空格分割每一行
        words = line.split()
        # 如果有四个以上的元素，则表示有文字信息
        if len(words) >= 5:
            # 获取文字的左上角坐标、宽度和高度
            text, x1, y1, x2, y2, page = words[0], int(words[1]), int(words[2]), int(words[3]), int(words[4]), int(words[5])
            max_height = max(max_height, y2 - y1)
    return max_height

def find_background_color(image):
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像转换为一维数组
    image_flat = np.reshape(image, (-1, 3))
    # 统计每个颜色值的出现次数
    unique_colors, counts = np.unique(image_flat, axis=0, return_counts=True)
    # 找到出现次数最多的颜色的索引
    most_common_color_index = np.argmax(counts)
    # 获取背景色
    return unique_colors[most_common_color_index]

def put_text(img: cv2.Mat, x: int, y: int, txt: str, height: int, txt_size: int, color=(0,0,0)):
    org = (x, y + height // 2)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = txt_size / 25
    thickness = int(txt_size / 30 * 2)
    lineType = cv2.LINE_AA
    color = tuple(map(int, color))
    cv2.putText(img, txt, org, fontFace, fontScale, color, thickness, lineType)
    show_img(img)
# def find_txt_color(image):


def find_icon(image):
    '''
        返回每个图标的标记框(x, y, w, h)
    '''
    background_color = find_background_color(image)
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用阈值化将背景颜色转换为纯黑色，图标转换为纯白色
    _, binary = cv2.threshold(gray, int(background_color[0]), 255, cv2.THRESH_BINARY)
    # 查找图标的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # 找到所有小框的边界框
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # 遍历边界框，找到大框的位置和大小
    x_min = min([x for x, _, _, _ in bounding_boxes])
    y_min = min([y for _, y, _, _ in bounding_boxes])
    x_max = max([x + w for x, _, w, _ in bounding_boxes])
    y_max = max([y + h for _, y, _, h in bounding_boxes])
    w = x_max - x_min
    h = y_max - y_min

    # 返回大框的坐标和大小
    return (x_min, y_min, w, h)
    # # 绘制矩形框标记图标
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # return image

def show_img(img: cv2.Mat):
    return
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_img(pth, img):
    cv2.imwrite(pth, img)

def overlay(img1, x_bias, y_bias, img2, width, heigth):
    for i in range(heigth):
        for j in range(width):
            if not np.all(img2[i, j]==0):
                img1[i+y_bias, j+x_bias] = img2[i, j, :3]
    return img1

def add_alpha_channel(src: np.ndarray) -> np.ndarray:
    if src.shape[2] != 3:
        raise ValueError("Input image must have 3 channels.")
    b_channel, g_channel, r_channel = cv2.split(src)
    alpha_channel = np.ones_like(b_channel) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    iterations = np.random.randint(1, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations)
    eroded = cv2.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap

def get_text(img: cv2.Mat):
    show_img(img)
    code = pytesseract.image_to_string(img, 'eng')
    print(code)
    return code

def text_bounds(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        regions, boxes = mser.detectRegions(gray)
        overlapThresh = 0.3
        # 定义NMS的函数
        def non_max_suppression_fast(boxes, overlapThresh):
            # 如果没有框，返回空列表
            if len(boxes) == 0:
                return []
            # 如果框是整数，转换为浮点数
            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")
            # 初始化选中的索引列表
            pick = []
            # 获取所有框的坐标
            x1 = boxes[:,0]
            y1 = boxes[:,1]
            x2 = boxes[:,0] + boxes[:,2]
            y2 = boxes[:,1] + boxes[:,3]
            # 计算所有框的面积并按y2升序排序，获取排序后的索引
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)
            # 循环处理剩余的索引
            while len(idxs) > 0:
                # 获取最后一个索引（y2最大的框）并加入选中列表
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)
                # 找到剩余框与当前框的最大坐标和最小坐标
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])
                # 计算重叠区域的宽度和高度
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                # 计算重叠区域的面积比率
                overlap = (w * h) / area[idxs[:last]]
                # 删除重叠比率大于阈值的索引
                idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
            # 返回选中的框，转换为整数类型
            return boxes[pick].astype("int")
        final_boxes = non_max_suppression_fast(boxes, overlapThresh)
        _x, _y = 0, 0
        for item in final_boxes:
            x, y, w, h = item
            _x = max(_x, x + w)
            _y = max(_y, y + h)
        return int(_x), int(_y)

if __name__ == '__main__':
    input_path = "D:\HongMeng\dsl\\tmp\\7.png"
    output_path = 'output.png'

    img = cv2.imread(input_path)
    # color = get_txt_color(img)
    # # print(color)
    # text = pytesseract.image_to_boxes(img, lang='eng')
    # # 按行分割字符串
    # lines = text.split('\n')
    # # 遍历每一行
    # color_dict = defaultdict(int)
    # for line in lines:
    #     # 按空格分割每一行
    #     words = line.split()
    #     # 如果有四个以上的元素，则表示有文字信息
    #     if len(words) >= 5:
    #         # 获取文字的左上角坐标、宽度和高度
    #         text, x1, y1, x2, y2, page = words[0], int(words[1]), int(words[2]), int(words[3]), int(words[4]), int(words[5])
    #         # 在图片上绘制矩形框
    #         colors = [img[y1 + 2, x1 + 2], img[y2 - 2, x1 + 2], img[y1 + 2, x2 - 2], img[y2 - 2, x2 - 2]]
    #         for c in colors:
    #             tc = tuple(map(int, c))
    #             color_dict[tc] += 1
    # color = max(color_dict, key=color_dict.get)
    # color = tuple(map(int,get_txt_color(img)))
    color = get_txt_color(img)
    print(color)
    x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
    color = tuple(map(int,color))
    # print(color)
    cv2.rectangle(img, (x1+3, y1 + 3), (x2 - 3, y2 - 3), color, 2)
    show_img(img)
    # # 转换图片为BGR模式
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # 获取(100, 100)坐标处的像素点的颜色信息
    # color = img[100, 100]
    # txt = get_text(input)
    # color = get_txt_color(input)
    # # put_text(input, 0, 0, txt, input.shape[0], (0, 0, 0))
    # put_text(input, 0, 0, txt, input.shape[0], tuple([int(x) for x in color]))
    # h, w, _ = input.shape
    # print(input.shape)
    # model_name = "isnet-general-use"
    # o = input.copy()
    # bg = find_background_color(input)
    # session = new_session(model_name)
    # output = remove(input, session=session, post_process_mask=True)
    # # output = remove(output, session=session)
    # show_img(output)
    # o = overlay(o, 10,10,output[0:h-10, 0:w-10], w-10, h-10)
    # cv2.imwrite(output_path, o)
    # trimap = gen_trimap(input)


    # widget = add_alpha_channel(input)

    