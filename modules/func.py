import random
from strenum import StrEnum
from .cv_helper import find_icon

import numpy as np
import cv2

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
    return final_boxes

def add_alpha_channel(color):
    # Create an alpha channel with all values set to 255 (fully opaque)
    alpha_channel = np.ones(1, dtype=np.uint8) * 255

    # Merge the R, G, B, and alpha channels into a single color
    return np.concatenate((color, alpha_channel), axis=None)

def cut(img: cv2.Mat, background) -> cv2.Mat:
    if img.shape[2] == 4:
        background = add_alpha_channel(background)
    h, w = img.shape[:2]
    img[:,w//2:] = background
    return img

def null(img: cv2.Mat, background) -> cv2.Mat:
    # 获取图片的宽度和高度
    height, width = img.shape[:2]
    all_color = img[5, 5]
    img[:,:] = all_color
    # 设置字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    # 计算文字的位置
    text_size = cv2.getTextSize("null", font, 1, 2)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    # 在图片上写文字
    cv2.putText(img, "null", (x, y), font, 1, color, 2)
    return img

def nochange(img: cv2.Mat, background) -> cv2.Mat:
    return img

def high_saturation(img: cv2.Mat, background) ->cv2.Mat:
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 增加饱和度
    s = hsv_image[:, :, 1]
    s = np.clip(s * 3, 0, 255)
    hsv_image[..., 1] = s

    # 将图像从HSV颜色空间转换回BGR颜色空间
    result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return result

def snow(img: cv2.Mat, background) ->cv2.Mat:
    a = random.random()
    h, w, c = img.shape
    if a < 0.5:
        # 生成一个随机的绿色通道的值，范围是128到255
        green = int(random.randint(128, 255))
        # 生成一个随机的红色通道的值，范围是0到127
        red = int(random.randint(0, 127))
        # 生成一个随机的蓝色通道的值，范围是0到127
        blue = int(random.randint(0, 127))
        for i in range(h):
            for j in range(w):
                if c == 3:
                    img[i, j] = [blue, green, red]
                else:
                    img[i, j] = [blue, green, red, 255]
                return img
    else:
        for i in range(h):
            for j in range(w):
                g = np.random.randint(50, 256)
                b = np.random.randint(g, 256)
                r = np.random.randint(g, 256)
                if c == 3:
                    img[i, j] = [b,g,r]
                else:
                    img[i, j] = [b, g, r, 255]
        return img

def distortion(img: cv2.Mat, background) ->cv2.Mat:
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.addWeighted(img, 0.5, img, 0.5, 50)
    # noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
    # 对Mat对象和噪声数组进行加法运算，得到一个新的Mat对象，表示添加了噪声的图片
    # img_noise = cv2.add(img, noise, dtype=cv2.CV_8U)
    return img

def occlusion(img: cv2.Mat, background) ->cv2.Mat:
    if img.shape[2] == 4:
        background = add_alpha_channel(background)
    h, w = img.shape[:2]
    r = np.random.uniform(0.3, 0.7)
    img[int(r*h):,:] = background
    return img

def imageloaderror(img: cv2.Mat, background) -> cv2.Mat:
    h, w = img.shape[:2]
    errorimg = cv2.imread('D:\HongMeng\dsl\Assets\\3.png')
    errorimg = cv2.resize(errorimg, (w, h) )
    return errorimg

def highSaturation(img: cv2.Mat, background) -> cv2.Mat:
    # 将图像从BGR颜色空间转换为HSV颜色空间
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 增加饱和度
    s = hsv_image[:, :, 1]
    s = np.clip(s * 4, 0, 255)
    hsv_image[..., 1] = s

    # 将图像从HSV颜色空间转换回BGR颜色空间
    result = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return result

def lowContrast(img: cv2.Mat, background) -> cv2.Mat:
    return img // 4

# if __name__ == "__main__":
    # img = cv2.imread('2.jpg')
    # color = img[800, 400]
    # img[:] = color
    # cv2.imshow('as', img)
    # cv2.waitKey(0)