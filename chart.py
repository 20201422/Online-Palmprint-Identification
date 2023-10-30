# @ClassName   Chart
# @Author  24
# @Date    2023/10/4 15:04
# @Version 1.0.0
# freedom is the oxygen of the soul.

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Arial Unicode MS']  # 设置中文支持


# 绘制汉明距离图
def get_hamming_distance_chart(genuine, imposter):
    # 计算真假匹配分布中的样本数n_genuine和n_imposter
    n_genuine = len(genuine)
    n_imposter = len(imposter)

    # 将横坐标分为500份
    x = np.linspace(0, 0.5, 500)

    # 统计个数
    genuine_y = np.zeros(500)
    imposter_y = np.zeros(500)
    for i in genuine:
        i = int(i * 1000)
        genuine_y[i] = genuine_y[i] + 1
    for i in imposter:
        i = int(i * 1000)
        imposter_y[i] = imposter_y[i] + 1

    genuine_y = genuine_y / n_genuine
    imposter_y = imposter_y / n_imposter

    plt.figure(0)
    plt.plot(x, genuine_y)
    plt.plot(x, imposter_y)
    plt.xlim(0, 0.5)

    # plt.hist(genuine, bins=50, density=True, alpha=0.5)
    # plt.hist(imposter, bins=50, density=True, alpha=0.5)

    plt.legend(['Genuine', 'Imposter'])
    plt.xlabel('Hamming distance')
    plt.ylabel('Percentage(%)')
    plt.savefig('Chart/hamming_distance_chart.svg')
    plt.clf()

    # 得到阈值
    get_threshold(x, genuine_y, imposter_y)


# 得到阈值
def get_threshold(x, genuine_y, imposter_y):
    # 找到genuine_y大于imposter_y的点的横坐标
    x1 = np.around(x[genuine_y > imposter_y], 2)
    # 找到imposter_y大于genuine_y的点的横坐标
    x2 = np.around(x[imposter_y > genuine_y], 2)

    # 求交集获取交点横坐标
    intersect_x = np.intersect1d(x1, x2)
    if len(intersect_x) > 0:
        print("Intersection point x coordinate:", intersect_x)
    else:
        print("No intersection point")


# 绘制ROC曲线图
def get_receiver_operating_characteristic(genuine, imposter):
    # 计算真假匹配分布中的样本数n_genuine和n_imposter
    n_genuine = len(genuine)
    n_imposter = len(imposter)

    # 计算假匹配率FAR和真匹配率GAR
    false_acceptance_rate = [0]
    genuine_acceptance_rate = [1]
    equal_error_rate = 0
    min_diff = 1000

    for t in np.arange(0.3, 0.5, 0.001):
        far = (sum(imposter <= t) / n_imposter)
        frr = (sum(genuine >= t) / n_genuine)
        gar = (sum(genuine <= t) / n_genuine)

        false_acceptance_rate.append(far * 100)
        genuine_acceptance_rate.append(gar * 100)

        # 找到等错误率ERR
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            equal_error_rate = far * 100

    print("Equal error rate:", equal_error_rate)

    # 画ROC曲线
    plt.plot(false_acceptance_rate, genuine_acceptance_rate)
    plt.ylim(90, 101)
    plt.xscale('log')
    plt.xlabel('False Acceptance Rate(%)')
    plt.ylabel('Genuine Acceptance Rate(%)')
    plt.savefig('Chart/roc.svg')


# 图表
def chart(genuine, imposter):
    get_hamming_distance_chart(genuine, imposter)
    get_receiver_operating_characteristic(genuine, imposter)


if __name__ == '__main__':
    genuine, imposter = np.load('genuine.npy'), np.load('imposter.npy')
    get_hamming_distance_chart(genuine, imposter)
    get_receiver_operating_characteristic(genuine, imposter)


#    may the force be with you.
#    @ClassName   Chart
#    Created by 24 on 2023/10/4.
