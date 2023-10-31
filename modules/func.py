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

def cut(img: cv2.Mat, background) -> cv2.Mat:
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


# if __name__ == "__main__":
    # img = cv2.imread('2.jpg')
    # color = img[800, 400]
    # img[:] = color
    # cv2.imshow('as', img)
    # cv2.waitKey(0)