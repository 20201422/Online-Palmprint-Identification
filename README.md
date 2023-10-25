# Online Palmprint Identification论文复现

## 运行环境：
Python 3.11

## 目录说明：
PolyU_Palm-point_600：存放原始掌纹采集图像

PalmROI：存放ROI图像

Gabor：存放每个掌纹图像的实部图像和虚部图像

Chart：存放运行的结果图表

## 文件说明：
### main.py
程序入口，运行main.py文件即可

    # 预处理图像 调用preprocess.py
    main_preprocess()

    # 提取特征 调用feature_extraction.py
    main_feature_extraction()

    # 匹配图像 调用match.py 返回匹配结果
    main_match()

    # 存入文件 将匹配结果存入genuine.npy和imposter.npy
    save_file(genuine_match_score, imposter_match_score)

    # 读取文件 读取genuine.npy和imposter.npy保存的匹配结果
    read_file()

### preprocess.py
图像预处理

    # 读取灰度图像 返回图像
    get_image(img_name)

    # 二值化图像 返回阈值化后的图像和所用的阈值
    get_binary_image(img)

    # 二值化图像转换为轮廓图 返回图像和手掌轮廓的坐标数组
    get_gaps_image(img, thresh)

    # 获取坐标点 返回两个坐标
    get_coordinate_point(contours, img.shape[0], img.shape[1])
    
    # 旋转图像 返回旋转后的新图像和两个旋转后的新的坐标点
    image_rotation(point_a, point_b, img)
    
    # 获取子图像 保存ROI图像到PalmROI目录
    get_sub_image(point_a, point_b, rotated, img_name)

    # 预处理图像 本文件的方法运行集合
    preprocess(img_name)

### feature_extraction.py
特征提取

    # 读取灰度图像 返回图像
    get_image(img_name)

    # 构建Gabor滤波器 返回滤波器
    gabor_filter(theta, u, sigma, n)

    # 滤波提取特征 返回滤波后的图像
    gabor_process(img)

    # 分离实部和虚部 返回实部和虚部
    separate_the_real_and_imaginary(feat)

    # 二值化 返回二值化图像
    binarization(img)

    # 下采样 128*128 -> 32*32 返回下采样图像
    downsampling(img)

    # 显示结果 用于打印提取结果
    print_image(real, imag, real_name, imag_name, file_name)

    # 特征提取 本文件的方法运行集合 保存滤波结果到Gabor目录
    feature_extraction(img_name)

### match.py
图像匹配

    # 读取图像 返回图像
    get_image(img_name, img_type)

    # 得到掩膜 返回掩膜图像
    get_mask_image(img_name)

    # 图像移动 返回移动后的图像
    move_image(img, x, y)

    # 图像裁剪 返回裁剪后的图像
    image_cropping(img, x, y)

    # 计算相似度 返回汉明距离
    cal_similarity(img1_real, img1_imag, img1_mask, img2_real, img2_imag, img2_mask, x, y)

    # 图像比较 本文件的方法运行集合
    match(img1_name, img2_name)

### chart.py
图表

    # 绘制汉明距离图 保存图表到Chart目录
    get_hamming_distance_chart(genuine, imposter)

    # 得到阈值 直接打印阈值
    get_threshold(x, genuine_y, imposter_y)

    # 绘制ROC曲线图 保存图表到Chart目录
    get_receiver_operating_characteristic(genuine, imposter)

    # 图表 本文件的方法运行集合
    chart(genuine, imposter)

### genuine.npy
保存类内匹配的匹配分数
### imposter.npy
保存类间匹配的匹配分数