# @ClassName   match
# @Author  24
# @Date    2023/10/3 15:49
# @Version 1.0.0
# freedom is the oxygen of the soul.

import cv2
import numpy as np
from skimage.morphology import opening


# 读取图像
def get_image(img_name, img_type):
    img = cv2.imread(img_name + '/Gabor_' + img_type + '_' + img_name.split("/")[1] + '.bmp', 0)
    if img is None:
        print('Failed to read the image!')
    return img


# 得到掩膜
def get_mask_image(img_name):
    # 获取图像
    img = cv2.imread('PalmROI/Cropped_' + img_name.split("/")[1] + '.bmp', 0)
    if img is None:
        print('Failed to read the image!')

    # 对掌纹图像进行阈值二值化
    thresh = 128
    binary_img = img > thresh

    # 对二值化图像进行开运算,去除一些噪声
    opening_img = opening(binary_img)

    # 计算开运算后图像中非零像素所在的位置,这些位置就是掌纹区域
    palm_points = np.where(opening_img > 0)

    # 生成掩膜,掌纹区域设置为1,其他设置为0
    mask = np.zeros_like(img)
    mask[palm_points] = 1

    return mask


# 图像移动
def move_image(img, x, y):
    m = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


# 图像裁剪
def image_cropping(img, x, y):
    img = img[abs(y):img.shape[0] - abs(y), abs(x):img.shape[1] - abs(x)]
    return img


# 计算相似度
def cal_similarity(img1_real, img1_imag, img1_mask, img2_real, img2_imag, img2_mask, x, y):
    # 只比较中间重复的部分
    img1_real = image_cropping(img1_real, x, y)
    img1_imag = image_cropping(img1_imag, x, y)
    img2_real = image_cropping(img2_real, x, y)
    img2_imag = image_cropping(img2_imag, x, y)

    diff_real = np.float32(img1_real != img2_real).sum() / img1_real.size
    diff_imag = np.float32(img1_imag != img2_imag).sum() / img1_imag.size
    # diff_mask = np.float32(img1_mask != img2_mask).sum() / img1_mask.size
    # print(diff_real, diff_imag)

    hamming_dist = (diff_real + diff_imag) / 2

    return hamming_dist


# 图像比较
def match(img1_name, img2_name):
    # 读取图像
    img1_real = get_image(img1_name, "real")
    img1_imag = get_image(img1_name, "imag")
    img2_real = get_image(img2_name, "real")
    img2_imag = get_image(img2_name, "imag")
    img1_mask = get_mask_image(img1_name)
    img2_mask = get_mask_image(img2_name)

    # 定义距离
    match_distance = 10000

    # 分别移动图像比较
    for i in range(-3, 4):
        for j in range(-3, 4):
            # 移动图像
            img1_real_move = move_image(img1_real, i, j)
            img1_imag_move = move_image(img1_imag, i, j)
            # 图像比较
            sim = cal_similarity(img1_real_move, img1_imag_move, img1_mask, img2_real, img2_imag, img2_mask, i, j)
            # 距离
            match_distance = min(sim, match_distance)

    return match_distance


if __name__ == '__main__':
    distance = match('Gabor/27_3', 'Gabor/27_5')
    print(distance)
    # cv2.imwrite('test.bmp', mask1)


#    may the force be with you.
#    @ClassName   match
#    Created by 24 on 2023/10/3.
