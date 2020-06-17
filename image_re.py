# from aip import AipOcr
import cv2
import numpy as np
from num_recon import templating


def img_devide():
    kernel = np.ones((5, 5), np.uint8)
    global num
    global boxes
    num = 0
    boxes = []
    for i in range(1):
        path = input('Please input the filename:')
        # 缩放图像
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        res = cv2.resize(img, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('res.png', res)
        equ = cv2.equalizeHist(res)
        cv2.imwrite('equ.png', equ)
        _, thres = cv2.threshold(
            equ, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imwrite('thres.png', thres)

        height, weight = thres.shape
        roi = thres[0:int(height*0.5), :]
        cv2.imwrite('roi.png', roi)

        opening = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(
            opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        img_postion, contours, _ = cv2.findContours(
            opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=len, reverse=True)
        for i in range(len(contours)):
            max_contours = contours[i]
            epsilon = 0.015 * cv2.arcLength(max_contours, True)
            approx = cv2.approxPolyDP(max_contours, epsilon, True)
            brims = len(approx)
            perimeter = cv2.arcLength(max_contours, True)
            area = cv2.contourArea(max_contours)

            # 挑选边框 选择目标区域
            if (brims >= 4 and (area/perimeter) > 50):

                # print(brims, area/perimeter)
                rect = cv2.minAreaRect(max_contours)
                box = np.int0(cv2.boxPoints(rect))
                boxes.append(box)
                cv2.drawContours(opening, [box], -1, (128, 132, 125), 5)
                cv2.imwrite('opening1.png', opening)

                num = num + 1
                if num == 2:
                    num = 0
                    break
        # 保存识别的区域
        for i in range(2):
            Xs = [i[0] for i in boxes[i]]
            Ys = [i[1] for i in boxes[i]]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            # print(x1, x2, y1, y2)
            roi_ = res[y1:y2, x1:x2]
            cv2.imwrite('roi' + str(i) + '.png', roi_)

        # 遍历数字图像，读出内容
        for i in ['roi0.png', 'roi1.png']:
            img_0 = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

            roi_h, roi_w = img_0.shape
            img_0 = img_0[int(roi_h*0.4):int(roi_h*0.8),
                          int(roi_w*0.1):int(roi_w*0.93)]
            cv2.imwrite('num_roi.png', img_0)

            blurred = cv2.GaussianBlur(img_0, (1, 1), 0)
            _, roi_thres = cv2.threshold(
                blurred, 20, 255, cv2.THRESH_BINARY)
            cv2.imwrite('roi_thres.png', roi_thres)

            kernel_1 = np.ones((3, 3), np.uint8)
            kernel_2 = np.ones((1, 1), np.uint8)
            roi_close = cv2.morphologyEx(cv2.GaussianBlur(
                roi_thres.copy(), (1, 1), 0), cv2.MORPH_CLOSE, kernel_2, iterations=1)
            cv2.imwrite('roi_close.png', roi_close)
            kernel_3 = np.ones((2, 2), np.uint8)
            roi_erosion = cv2.erode(roi_close, kernel_3, iterations=1)
            cv2.imwrite('roi_erosion.png', roi_erosion)
            # 数字识别
            templating('roi_erosion.png')


if __name__ == '__main__':
    img_devide()
