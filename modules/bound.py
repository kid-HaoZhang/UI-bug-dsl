# import numpy as np
# import cv2

# #定义全局变量
# n = 0    #定义鼠标按下的次数
# ix = 0   # x,y 坐标的临时存储
# iy = 0
# rect = (0,0,0,0) #前景区域

# #鼠标回调函数
# def draw_rectangle(event,x,y,flags,param):
#     global n,ix,iy,rect
#     if event==cv2.EVENT_LBUTTONDOWN :
#         if n == 0:    #首次按下保存坐标值
#             n+=1
#             ix,iy = x,y
#         else:        #第二次按下显示矩形
#             n+=1
#             rect = (ix,iy,(x-ix),(y-iy))#前景区域
            
# #读取图像
# img = cv2.imread('D:\HongMeng\dsl\\tmp\\4.png')
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (0, 0, img.shape[0], img.shape[1])
# #前景提取
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# #显示图像
# cv2.imshow("img",img)
# cv2.waitKey()
# cv2.destroyAllWindows()
import cv2
import numpy as np

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