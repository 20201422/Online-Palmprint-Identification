# @ClassName   feature_extraction
# @Author  24
# @Date    2023/10/3 15:02
# @Version 1.0.0
# freedom is the oxygen of the soul.

import cv2
import numpy as np
from scipy import signal
import skimage.measure
import matplotlib.pyplot as plt
import os


# 读取灰度图像
def get_image(img_name):
    img = cv2.imread(img_name, 0)
    if img is None:
        print('Failed to read the image!')
    return img


# 构建Gabor滤波器
def gabor_filter(theta, u, sigma, n):
    g = np.zeros((2 * n + 1, 2 * n + 1), dtype=complex)

    for x in range(-n, n + 1):
        for y in range(-n, n + 1):
            g[x + n, y + n] = ((1 / (2 * np.pi * sigma ** 2)) *
                               np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2) +
                                      2 * np.pi * 1j * (u * np.cos(theta) * x + u * np.sin(theta) * y)))

    g = g - np.mean(np.mean(g))

    return g


# 滤波提取特征
def gabor_process(img):
    feat = signal.convolve2d(img, gabor_filter(np.pi / 4, 0.0916, 5.6179, 8), mode='same')

    return feat


# 分离实部和虚部
def separate_the_real_and_imaginary(feat):
    real = np.real(feat)
    imag = np.imag(feat)

    return real, imag


# 二值化
def binarization(img):
    img[img >= 0] = 255
    img[img < 0] = 0

    return img


# 下采样 128*128 -> 32*32
def downsampling(img):
    img_downscaled = skimage.measure.block_reduce(img, (4, 4), np.mean)
    # 设定阈值,这里取128
    threshold = 128
    # 二值化
    img_binary = np.where(img_downscaled >= threshold, 255, 0)

    return img_binary


# 显示结果
def print_image(real, imag, real_name, imag_name, file_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(real, cmap='gray')
    plt.title(real_name + ' Part')
    plt.subplot(1, 2, 2), plt.imshow(imag, cmap='gray')
    plt.title(imag_name + ' Part')
    plt.savefig('/Users/24/Desktop/研究生/论文复现/Online Palmprint Identification/' + file_name + '.png')
    plt.show()

    # cv2.imwrite('/Users/24/Desktop/研究生/论文复现/Online Palmprint Identification/' + real_name + '.bmp', real)
    # cv2.imwrite('/Users/24/Desktop/研究生/论文复现/Online Palmprint Identification/' + imag_name + '.bmp', imag)


# 特征提取
def feature_extraction(img_name):
    # 读取灰度图像
    img = get_image(img_name)
    # 滤波提取特征
    feat = gabor_process(img)
    # 分离实部和虚部
    real, imag = separate_the_real_and_imaginary(feat)
    # 二值化
    real = binarization(real)
    imag = binarization(imag)

    # 下采样 128*128 -> 32*32
    real_binary = downsampling(real)
    imag_binary = downsampling(imag)

    # 显示结果
    # print_image(real, imag,"Real", "Imag", "binarization")
    # print_image(real_binary, imag_binary, "Real Downscaled", "Imag Downscaled", "downscaled")

    # 将滤波结果转换为图片格式并保存
    if not os.path.exists('Gabor/' + img_name.split("/")[1].split("_", 1)[1].split(".")[0]):
        os.mkdir('Gabor/' + img_name.split("/")[1].split("_", 1)[1].split(".")[0])
    cv2.imwrite('Gabor/' + img_name.split("/")[1].split("_", 1)[1].split(".")[0] + '/Gabor_real_'
                + img_name.split("/")[1].split("_", 1)[1], real_binary)
    cv2.imwrite('Gabor/' + img_name.split("/")[1].split("_", 1)[1].split(".")[0] + '/Gabor_imag_'
                + img_name.split("/")[1].split("_", 1)[1], imag_binary)


if __name__ == '__main__':
    feature_extraction('PalmROI/Cropped_00_1.bmp')


#    may the force be with you.
#    @ClassName   feature_extraction
#    Created by 24 on 2023/10/3.
