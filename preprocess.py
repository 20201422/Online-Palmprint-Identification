# @ClassName   preprocess
# @Author  24
# @Date    2023/10/2 19:43
# @Version 1.0.0
# freedom is the oxygen of the soul.

import cv2
import numpy as np
import math


# 读取灰度图像
def get_image(img_name):
    img = cv2.imread(img_name, 0)
    if img is None:
        print('Failed to read the image!')
    return img


# 二值化图像
def get_binary_image(img):
    # 使用高斯滤波进行平滑处理,去除噪声
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imwrite('/Users/24/Desktop/研究生/论文复现/Online Palmprint Identification/blur_image.bmp', blur)
    # 使用阈值二值化图像
    ret, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
    # 保存二值化图像
    # cv2.imwrite('/Users/24/Desktop/研究生/论文复现/Online Palmprint Identification/binary_image.bmp', thresh)
    return ret, thresh


# 二值化图像转换为轮廓图
def get_gaps_image(img, thresh):
    # 寻找所有轮廓 按逆时针方向存储轮廓点
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 计算每个轮廓的面积,找到最大的
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    max_cnt = contours[max_index]

    # 创建结果图
    contour_img = np.zeros(img.shape)

    # 只绘制最大轮廓
    cv2.drawContours(contour_img, [max_cnt], -1, (255, 255, 255), 1)

    # 保存结果
    # cv2.imwrite('/Users/24/Desktop/研究生/论文复现/Online Palmprint Identification/contours.bmp', contour_img)

    return contour_img, max_cnt


# 获取坐标点
def get_coordinate_point(contours, img_height, img_weight):
    # 定义食指与中指之间的坐标
    point_a = [0, 0]
    # 定义无名指与小拇指之间的坐标
    point_b = [0, 0]
    # 标记两个坐标是否已经获取
    flag_a, flag_b = 0, 0
    # 当前坐标的前一个（默认第一个（左上角））
    pre_x, pre_y = contours[0][0][0], contours[0][0][1]

    # 遍历每个轮廓找到第一个坐标
    for cnt in contours:
        # 获取轮廓上的坐标点
        for point in cnt:
            x, y = point[0], point[1]
            if flag_a == 0 and y < img_height / 2 and x < img_weight / 2:  # 如果还没有获取坐标A
                if pre_x <= x and pre_y <= y and point_a[0] <= x:  # 获取x和y均为非递减的最后一个坐标
                    point_a = [x, y]
                elif point_a[0] != contours[0][0][0]:     # 如果不是x和y均为非递减且坐标已经被更改 那么标记已经获取
                    flag_a = 1

            if flag_a == 1:   # 坐标已获取 终止循环
                break

            pre_x, pre_y = x, y

    # 当前坐标的前一个（默认第一个（左上角））
    pre_x, pre_y = contours[0][0][0], contours[0][0][1]
    # x为非递减y为非递增的子数组的个数
    count = 0

    # 倒着遍历每个轮廓找到第二个坐标
    for cnt in reversed(contours):
        # 获取轮廓上的坐标点
        for point in cnt:
            x, y = point[0], point[1]
            if flag_b == 0 and y > img_height / 2 and x < img_weight / 2:  # 如果还没有获取坐标A
                if pre_x <= x and point_b[0] <= x:  # 获取x为非递减的最后一个坐标
                    count += 1
                    if count > 27:  # 目的是排除前面短的连续的像素点
                        point_b = [x, y]
                elif point_b[0] != 0:  # 如果不是x和y均为非递减且坐标已经被更改 那么标记已经获取
                    flag_b = 1
                else:
                    count = 0

            if flag_b == 1:   # 坐标已获取 终止循环
                break

            pre_x, pre_y = x, y

    return point_a,  point_b


# 获取两条直线以及交点
def get_line(point_a, point_b):
    # 计算两个点的中点坐标
    x_mid = (point_a[0] + point_b[0]) / 2
    y_mid = (point_a[1] + point_b[1]) / 2
    # p_mid = (int(x_mid), int(y_mid))

    if point_b[0] != point_a[0]:
        # 计算两点所在直线的斜率
        k = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])

        # 计算过中点的垂线的斜率
        k1 = -1 / k

        # 计算y=kx+b的b
        b = point_b[1] - k * point_b[0]
        b1 = y_mid - k1 * x_mid

        return k, b, k1, b1, x_mid, y_mid
    else:
        return 0, 0, 0, 0, x_mid, y_mid


# 计算两直线的交点
def find_intersection(k, b, k_prime, b_prime):
    x = (b_prime - b) / (k - k_prime)
    y = k * x + b

    return x, y


# 计算直线上距离一点x的点的坐标
def get_point_distance(x1, y1, k, distance):
    x = x1 + distance * math.sqrt(1 / (k ** 2 + 1))
    y = k * (x - x1) + y1

    return int(x), int(y)


# 获取新坐标
def get_new_point(point_a, point_b, rot_mat):
    # 转换为同质坐标
    point_a_h = np.array([point_a[0], point_a[1], 1])
    point_b_h = np.array([point_b[0], point_b[1], 1])

    # 利用旋转矩阵计算新坐标
    new_point_a = rot_mat.dot(point_a_h)
    new_point_b = rot_mat.dot(point_b_h)

    # 转换回笛卡尔坐标
    new_point_a = (int(new_point_a[0]), int(new_point_a[1]))
    new_point_b = (int(new_point_b[0]), int(new_point_b[1]))

    return new_point_a, new_point_b


# 旋转图像
def image_rotation(point_a, point_b, img):
    # 获取两条直线
    k, b, k1, b1, x_mid, y_mid = get_line(point_a, point_b)
    # 计算旋转角度 theta
    theta = np.arctan(k)
    # 旋转中心
    center = tuple(np.array(img.shape[1::-1]) / 2)
    # 旋转变换矩阵
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(theta), 1.0)
    # 执行旋转
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # 获取新坐标
    point_a, point_b = get_new_point(point_a, point_b, rot_mat)

    if k != 0:
        # 获取高、宽
        height, width = result.shape[:2]
        # 构造旋转矩阵,逆时针旋转90度
        if k > 0:
            m = cv2.getRotationMatrix2D((width / 2, height / 2), -90, 1)
        else:
            m = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
        # 进行仿射变换,得到旋转后的图像
        rotated = cv2.warpAffine(result, m, (width, height))
        # 获取新坐标
        point_a, point_b = get_new_point(point_a, point_b, m)

        # 显示结果
        # cv2.imshow('Rotated Image', rotated)
        # cv2.waitKey(0)
        # cv2.imwrite('/Users/24/Desktop/rotated.bmp', rotated)

        return rotated, point_a, point_b

    else:
        return result, point_a, point_b


# 获取子图像
def get_sub_image(point_a, point_b, img, img_name):
    # 获取两条直线
    k, b, k1, b1, x_mid, y_mid = get_line(point_a, point_b)

    # 计算图像中心坐标
    x = int(x_mid + 100)
    y = int(y_mid)

    # # 通过中心坐标获取子图像的四点坐标
    # x1_mid, y1_mid = find_intersection(k, b + 1000, k1, b1)
    # # 指定距离
    # distance = 64
    # # 计算目标点
    # x1, y1 = get_point_distance(x1_mid, y1_mid, k, distance)
    # x2, y2 = get_point_distance(x1_mid, y1_mid, k, -distance)
    # x2_mid, y2_mid = get_point_distance(x1_mid, y1_mid, k1, 2 * distance)
    # x3, y3 = get_point_distance(x2_mid, y2_mid, k, distance)
    # x4, y4 = get_point_distance(x2_mid, y2_mid, k, -distance)
    #
    # points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    # print(points)

    # 裁剪图像
    cropped = img[int(y - 64):int(y + 64), int(x - 64):int(x + 64)]
    # 保存结果
    cv2.imwrite('PalmROI/Cropped_' + img_name.split("/")[1].split("_", 1)[1], cropped)


# 预处理图像
def preprocess(img_name):

    # 读取灰度图像
    img = get_image(img_name)
    # 二值化图像
    ret, thresh = get_binary_image(img)
    # 二值化图像转换为轮廓图
    contour_img, contours = get_gaps_image(img, thresh)
    # 获取坐标点
    point_a, point_b = get_coordinate_point(contours, img.shape[0], img.shape[1])
    # 旋转图像
    rotated, point_a, point_b = image_rotation(point_a, point_b, img)
    # 获取子图像
    get_sub_image(point_a, point_b, rotated, img_name)


if __name__ == '__main__':
    preprocess('PolyU_Palm-print_600/PolyU_00_1.bmp')


#    may the force be with you.
#    @ClassName   preprocess
#    Created by 24 on 2023/10/2.
