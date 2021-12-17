import cv2 as cv
import pytesseract
from PIL import Image


def recognize_text(image):
    # 边缘保留滤波  去噪
    blur = cv.pyrMeanShiftFiltering(image, sp=8, sr=60)
    cv.imshow('dst', blur)
    # 灰度图像
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # 二值化  设置阈值  自适应阈值的话 黄色的4会提取不出来
    ret, binary = cv.threshold(gray, 185, 255, cv.THRESH_BINARY_INV)
    print(f'二值化设置的阈值：{ret}')
    cv.imshow('binary', binary)
    # 逻辑运算  让背景为白色  字体为黑  便于识别
    cv.bitwise_not(binary, binary)
    cv.imshow('bg_image', binary)
    # 识别
    test_message = Image.fromarray(binary)
    text = pytesseract.image_to_string(test_message, lang='chi_sim')
    print('识别结果：', text)


src = cv.imread(r'bjtu_cc_vertification\\0.jpg')
cv.imshow('input image', src)
recognize_text(src)
cv.waitKey(0)
cv.destroyAllWindows()
