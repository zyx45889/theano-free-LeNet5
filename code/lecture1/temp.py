import cv2
import numpy as np

add = 'C:/Users/86133/Desktop/XBMI/picture/binary/'
photo_num=19
for i in range(photo_num):
    image= cv2.imread(add+str(i+1)+'.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    cv2.imwrite(add+str(i+1+100)+'.jpg', binary)
