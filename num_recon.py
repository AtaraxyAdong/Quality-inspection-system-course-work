import cv2
import numpy as np
from matplotlib import pyplot as plt


def templating(img_path):
    template_path = ['temple1_E.png', 'temple2_E.png',
                     'temple1_r.png', 'temple2_r.png',
                     'temple1_-.png',
                     'temple2_9.png', 'temple1_9.png',
                     'temple3_2.png', 'temple1_2.png', 'temple4_2.png',  # 'temple2_2.png',
                     'temple1_3.png', 'temple2_3.png',
                     'temple1_5.png', 'temple2_5.png',
                     'temple1_6.png', 'temple2_6.png',
                     'temple1_7.png', 'temple2_7.png',
                     'temple3_6.png',  'temple3_0.png', 'temple3_8.png', 'temple2_8.png', 'temple1_8.png', 'temple4_8.png', 'temple3_0.png',
                     'temple1_0.png', 'temple2_0.png',
                     'temple1_1.png',  # 'temple2_1.png',
                     'temple3_..png', 'temple1_..png', 'temple2_..png', 'temple4_..png'
                     ]

    # num_img = cv2.imread('roi_thres.png', cv2.IMREAD_GRAYSCALE)
    num_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    num_x = []
    for path in template_path:
        # path = 'temple_2.png'
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        num = list(path)[-5]
        h, w = template.shape
        res = cv2.matchTemplate(num_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.75

        # 获取坐标信息
        loc = np.where(res >= threshold)
        pos_x = (loc[::-1][0])
        pos_y = (loc[::-1][1])

        # 排除无匹配情况
        if len(pos_y) == 0:
            continue

        # 计算y的均值
        sum = 0
        for i in pos_y:
            sum = sum + i
        pos_y_mean = int(sum / len(pos_y))

        # 删除重复x坐标值
        B = set(pos_x)
        C = list(B)

        # 删除数值相近的x坐标
        for i in C:
            for j in B:
                if (abs(i-j) < 5) and (abs(i-j) > 0):
                    C.remove(j)

        # 更新坐标
        pos_x = C

        for x in pos_x:
            # 绘制最小外接矩形
            # cv2.rectangle(
            #     num_img, (x, pos_y_mean), (x + w, pos_y_mean + h), (255, 255, 255), 1)

            # 匹配区域设置为白色
            white = np.ones((h, w), dtype=np.uint8) * 255
            num_img[pos_y_mean:(pos_y_mean + h), x:(x + w)] = white
            num_x.append([x, num])

    # 进行坐标排序，输出数字
    num_x_sorted = sorted(num_x, key=(lambda x: x[0]))
    out = ''
    for i in num_x_sorted:
        for j in i[1]:
            out = out + str(j)
    print('电表显示为:', out)


# templating(img_path)
