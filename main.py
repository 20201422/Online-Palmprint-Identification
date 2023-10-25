# @ClassName   main
# @Author  24
# @Date    2023/10/2 19:35
# @Version 1.0.0
# freedom is the oxygen of the soul.

import preprocess
import feature_extraction
import match
import chart

import glob
import os
import itertools
import re
import numpy as np
from tqdm import tqdm


# 预处理图像
def main_preprocess():
    bmp_files = glob.glob('PolyU_Palm-print_600/*.bmp')
    for file in tqdm(bmp_files, desc="图像预处理", unit="张"):
        preprocess.preprocess('PolyU_Palm-print_600/' + os.path.basename(file))  # 预处理图像


# 提取特征
def main_feature_extraction():
    bmp_files = glob.glob('PalmROI/*.bmp')
    for file in tqdm(bmp_files, desc="特征提取", unit="张"):
        feature_extraction.feature_extraction('PalmROI/' + os.path.basename(file))  # 提取特征


# 匹配图像
def main_match():
    genuine_match_score, imposter_match_score = [], []
    # 编译正则表达式,匹配"两位数字_一位数字"文件名
    pattern = re.compile(r'\d{2}_\d')

    files_gen = (f for f in os.listdir('Gabor'))

    for file1, file2 in tqdm(itertools.combinations(files_gen, 2), desc="图像匹配", unit="次"):
        # 检查两个文件名是否都匹配格式
        if not pattern.match(os.path.basename(file1)) or not pattern.match(os.path.basename(file2)):
            continue

        # 匹配图像
        distance = match.match('Gabor/' + file1, 'Gabor/' + file2)

        # 如果是同一只手
        if os.path.basename(file1).split('_')[0] == os.path.basename(file2).split('_')[0]:
            # if distance > 0.35:
            #     print(file1, file2, distance)
            genuine_match_score.append(distance)  # 保存到数组
        # 如果不是同一只手
        else:
            imposter_match_score.append(distance)  # 保存到数组

    # print(genuine_match_score)
    # print("\n\n\n\n")
    # print(imposter_match_score)

    return genuine_match_score, imposter_match_score


# 存入文件
def save_file(genuine_match_score, imposter_match_score):
    np.save('genuine.npy', genuine_match_score)
    np.save('imposter.npy', imposter_match_score)


# 读取文件
def read_file():
    return np.load('genuine.npy'), np.load('imposter.npy')


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # # 预处理图像
    main_preprocess()

    # 提取特征
    main_feature_extraction()

    # 匹配图像
    genuine, imposter = main_match()
    print("完成图像匹配")

    # 匹配结果存入到文件
    save_file(genuine, imposter)

    # 读取数组文件
    genuine, imposter = read_file()

    # 绘制图表
    chart.chart(genuine, imposter)
    print("完成图表绘制")


#    may the force be with you.
#    @ClassName   main
#    Created by 24 on 2023/10/2.
