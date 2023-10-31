from rembg import remove
import cv2
import numpy as np
from PIL import Image

def remove_bg(img: cv2.Mat):
    _, img_bytes = cv2.imencode('.png', img) 
    img_bytes = img_bytes.tobytes()
    result_bytes = remove(img_bytes)
    result_np = np.frombuffer(result_bytes, np.uint8)
    result_mat = cv2.imdecode(result_np, cv2.IMREAD_UNCHANGED)
    return result_mat

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
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlay(img1, img2, x_bias, y_bias):
    h, w = img2.shape[:2]
    for i in range(h):
        for j in range(w):
            if not np.all(img2[i, j]==0):
                img1[i+y_bias, j+x_bias] = img2[i, j, :3]
    return img1

if __name__ == '__main__':
    # img = cv2.imread('D:\HongMeng\dsl\\tmp\\3.png', cv2.IMREAD_UNCHANGED)
    # hei, wid, _ = img.shape
    # # print(wid, hei)
    # widget_img = remove_bg(img)
    # # print(widget_img)
    # # mask = np.where((widget_img [:,:,0] == 0) & (widget_img [:,:,1] == 0) & (widget_img [:,:,2] == 0), 0, 255).astype(widget_img.dtype)
    # overlayd = img.copy()
    # # tran = cv2.bitwise_and(widget_img, widget_img, mask=mask)
    # # print(tran.shape)
    # overlayd = overlay(overlayd, widget_img[0:hei-20, 0:wid-20], 20, 20)
    # # overlay[20:hei, 20:wid] = tran[0:hei-20, 0:wid-20]
    # show_img(overlayd)
    input_path = 'D:\HongMeng\dsl\\tmp\\4.png'
    output_path = 'output.png'

    input = cv2.imread(input_path)
    h, w, _ = input.shape
    print(input.shape)
    output = remove(input)
    output = remove(output)
    output = remove(output)
    # 创建一个空白的掩码，用于存放output的透明度
    # mask = np.zeros((h, w), dtype=np.uint8)
    # mask[40:h, 40:w] = output[0:h-40, 0:w-40, 3]
    # mask = mask.astype(bool)
    # print(mask.shape)
    # print(output.shape)
    # # 将output的前三个通道（颜色）赋值给input的对应位置，且只覆盖非透明的部分
    # input[40:h, 40:w][mask[0:h-40, 0:w-40]] = output[0:h-40, 0:w-40][mask[0:h-40, 0:w-40], :3]
    o = overlay(input, output[0:h-20, 0:w-20], 20, 20)
    cv2.imwrite(output_path, o)
    