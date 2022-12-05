import argparse
import imutils
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

frame = cv2.imread("../Automatic_correction_image/aruco2_x.png")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters_create()
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

#cv2.imshow("Image", frame_markers)

#print in image the coordinates of the corners of the markers
for i in range(len(corners)):
    c = corners[i][0]
    cv2.line(frame, tuple(c[0].astype('int32')), tuple(c[1].astype('int32')), (255,0,255), 5)
    cv2.line(frame, tuple(c[1].astype('int32')), tuple(c[2].astype('int32')), (255,0,255), 5)
    cv2.line(frame, tuple(c[2].astype('int32')), tuple(c[3].astype('int32')), (255,0,255), 5)
    cv2.line(frame, tuple(c[3].astype('int32')), tuple(c[0].astype('int32')), (255,0,255), 5)
    cv2.putText(frame, str(c[0]), tuple(c[0].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.putText(frame, str(c[1]), tuple(c[1].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.putText(frame, str(c[2]), tuple(c[2].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.putText(frame, str(c[3]), tuple(c[3].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    #draw circle in left down corner of the marker with the id number equal to 1
    if ids[i] == 1:
        cv2.circle(frame, tuple(c[3].astype('int32')), 5, (255,0,0), -1)
        x1,y1 = c[3].astype('int32')
    #draw circle in right down corner of the marker with the id number equal to 2
    if ids[i] == 2:
        cv2.circle(frame, tuple(c[2].astype('int32')), 5, (255,0,0), -1)
        x2,y2 = c[2].astype('int32')
    #draw circle in left up corner of the marker with the id number equal to 3
    if ids[i] == 3:
        cv2.circle(frame, tuple(c[0].astype('int32')), 5, (255,0,0), -1)
        x3,y3 = c[0].astype('int32')
    #draw circle in right up corner of the marker with the id number equal to 4
    if ids[i] == 4:
        cv2.circle(frame, tuple(c[1].astype('int32')), 5, (255,0,0), -1)
        x4,y4 = c[1].astype('int32')


## binarising image
gray_scale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
th1,img_bin = cv2.threshold(gray_scale,220,225,cv2.THRESH_BINARY)



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
cv2.imshow("Image horizontal lines", img_bin_h)
## detect vert lines
img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1v)  # bridge small gap in vert lines
img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)# kep ony vert lines by eroding everything else in vert direction
cv2.imshow("Image vertical lines", img_bin_v)
### function to fix image as binary
def fix(img):
    img[img>127]=255
    img[img<127]=0
    return img

img_bin_final = fix(fix(img_bin_h)|fix(img_bin_v))
finalKernel = np.ones((5,5), np.uint8)
# este dilate serve para dilatar as linhas para se conseguir detetar melhor a seguir
img_bin_final=cv2.dilate(img_bin_final,finalKernel,iterations=1)

ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

edges = cv2.Canny(img_bin_final,1,128)
cv2.imshow("Image edges", edges)


#contours
contours, hierarchy = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, (0,255,0), 3)

#show contours
cv2.imshow("Image contours", frame)


img_x = cv2.imread('../Automatic_correction_image/x.png',0)
edges = cv2.Canny(img_x,100,200)
cv2.imshow("Image x edges", edges)

'''
### skipping first two stats as background
for x,y,w,h,area in stats[2:]:
    # x cresce para a direita e o y para baixo
    if x>=x1 and x<=x2 and y>=y1 and y<=y3:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        rectangle = frame[y:y+h,x:x+w]
        plt.figure(figsize=(1,1))
        plt.imshow(rectangle)
        plt.show()
        
'''  

cv2.imshow("Image", frame)
        
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
plt.figure()
plt.imshow(frame_markers)
for i in range(len(ids)):
    c = corners[i][0]
    plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
plt.legend()
plt.show()

"""
