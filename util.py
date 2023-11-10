import random
import cv2
import numpy as np

def select_random_numbers(n):
    numbers = list(range(0, n))
    selected_numbers = []
    for i in range(0, len(numbers), 4):
        selected_numbers.append(random.choice(numbers[i:i+4]))
    return selected_numbers


def remove_background(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 创建一个掩码，将背景标记为黑色
    mask = cv2.createBackgroundSubtractorMOG2().apply(image)

    # 对掩码进行处理，将背景变为透明
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask[np.all(mask == [0, 0, 0], axis=2)] = [0, 0, 0]

    # 将掩码应用于原始图像，提取前景
    foreground = cv2.bitwise_and(image, mask)

    # 返回前景图像
    return foreground

if __name__ == "__main__":
    # print(select_random_numbers(6))
    # 示例用法
    # image_path = 'D:\HongMeng\dsl\\tmp\\4.png'
    # foreground_image = remove_background(image_path)
    # cv2.imshow('Foreground Image', foreground_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(select_random_numbers(5))