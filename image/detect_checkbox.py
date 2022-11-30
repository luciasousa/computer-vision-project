import cv2
from matplotlib import pyplot as plt
import numpy as np

image=cv2.imread('../Automatic_correction_image/test1.png')

### binarising image
gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
th1,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)

lineWidth = 7
lineMinWidth = 55
kernal1 = np.ones((lineWidth,lineWidth), np.uint8)
kernal1h = np.ones((1,lineWidth), np.uint8)
kernal1v = np.ones((lineWidth,1), np.uint8)

kernal6 = np.ones((lineMinWidth,lineMinWidth), np.uint8)
kernal6h = np.ones((1,lineMinWidth), np.uint8)
kernal6v = np.ones((lineMinWidth,1), np.uint8)

img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1h) # bridge small gap in horizonntal lines
img_bin_h = cv2.morphologyEx(img_bin_h, cv2.MORPH_OPEN, kernal6h) # kep ony horiz lines by eroding everything else in hor direction

## detect vert lines
img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1v)  # bridge small gap in vert lines
img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)# kep ony vert lines by eroding everything else in vert direction

### function to fix image as binary
def fix(img):
    img[img>127]=255
    img[img<127]=0
    return img

img_bin_final = fix(fix(img_bin_h)|fix(img_bin_v))

finalKernel = np.ones((5,5), np.uint8)
img_bin_final=cv2.dilate(img_bin_final,finalKernel,iterations=1)

ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

### skipping first two stats as background
for x,y,w,h,area in stats[2:]:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    #plot each rectangle
    rectangle = image[y:y+h,x:x+w]
    plt.figure(figsize=(1,1))
    plt.imshow(rectangle)
    plt.show()
    

cv2.imshow('image',image)
cv2.waitKey(0)