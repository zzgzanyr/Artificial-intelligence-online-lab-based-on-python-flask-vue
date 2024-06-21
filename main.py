# ———————— 实验四 ——————————————
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr

# 二值化
def binaryzation(img):
    maxi = float(img.max())
    mini = float(img.min())
    x = maxi - ((maxi - mini) / 2)
    ret, thresh = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
    return thresh

# 找到能够包围给定区块的最小矩形的左下角坐标(min(x),min(y))与右上角坐标(max(x),max(y))
def find_rectangle(contour):
    x = []
    y = []
    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])
    return [min(y), min(x), max(y), max(x)]

# 读取图片
img = cv2.imread(r"C:\Users\17927\Desktop\nimasile.jpeg")
cv2.imshow('initial image', img)
m = 400 * img.shape[0] / img.shape[1]
# 调整图像尺寸
img = cv2.resize(img, (400, int(m)), interpolation=cv2.INTER_AREA)
# 转化为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 结构元素(kernel)为半径为16的圆
r = 16
h = w = r * 2 + 1
kernel = np.zeros((h, w), np.uint8)
cv2.circle(kernel, (r, r), r, 1, -1)
# 开运算
open_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
# 顶帽
hat_img = cv2.absdiff(gray_img, open_img)
# 图像二值化
binary_img = binaryzation(hat_img)
# canny边缘检测
canny = cv2.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])
# 闭运算核
kernel = np.ones((5, 19), np.uint8)
# 闭运算
close_img = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
# 开运算
open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
# 更换结构元素
kernel = np.ones((11, 5), np.uint8)
# 开运算
open_img = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel)
# 提取轮廓
contours, hierarchy = cv2.findContours(open_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 存储每个块的矩形坐标
block = []
for c in contours:
    r = find_rectangle(c)
    block.append(r)

max_weight = 0
max_index = -1
# 遍历分割出来的各个区域
for i in range(len(block)):
    b = img[block[i][1]: block[i][3], block[i][0]: block[i][2]]
    # 转化为hsv颜色空间
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    # 蓝色下界
    lower = np.array([100, 50, 50])
    # 蓝色上界
    upper = np.array([140, 255, 255])
    # 利用掩膜进行筛选
    mask = cv2.inRange(hsv, lower, upper)
    # 计算当前区域的满足情况
    w1 = 0
    for m in mask:
        w1 += m / 255
    w2 = 0
    for n in w1:
        w2 += n
    if w2 > max_weight:
        max_index = i
        max_weight = w2
# 最可能为车牌的区域对应的矩形坐标
rect = block[max_index]
# 画框
cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

# 使用PaddleOCR识别车牌区域文字
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 初始化 PaddleOCR
license_plate = img[rect[1]:rect[3], rect[0]:rect[2]]
gray_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
result = ocr.ocr(gray_license_plate, cls=True)

# 打印识别出的文字
for line in result:
    for word in line:
        print("识别出的文字为:", word[1][0])

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
