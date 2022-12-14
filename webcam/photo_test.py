#programa que tira foto ao teste e depois faz o processamento
import argparse
import imutils
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    if ids is not None:
        length = len(ids)
        if length == 4:
            for i in range(len(corners)):
                c = corners[i][0]# c type = numpy.ndarray
                #cv2.line(frame, tuple(c[0].astype('int32')), tuple(c[1].astype('int32')), (255,0,255), 5)
                #cv2.line(frame, tuple(c[1].astype('int32')), tuple(c[2].astype('int32')), (255,0,255), 5)
                #cv2.line(frame, tuple(c[2].astype('int32')), tuple(c[3].astype('int32')), (255,0,255), 5)
                #cv2.line(frame, tuple(c[3].astype('int32')), tuple(c[0].astype('int32')), (255,0,255), 5)
                if ids[i] == 1:
                    cv2.circle(frame, tuple(c[3].astype('int32')), 5, (255,0,0), -1)
                    x1,y1 = c[3].astype('int32')
                if ids[i] == 2:
                    cv2.circle(frame, tuple(c[2].astype('int32')), 5, (255,0,0), -1)
                    x2,y2 = c[2].astype('int32')
                if ids[i] == 3:
                    cv2.circle(frame, tuple(c[0].astype('int32')), 5, (255,0,0), -1)
                    x3,y3 = c[0].astype('int32')
                if ids[i] == 5:
                    cv2.circle(frame, tuple(c[1].astype('int32')), 5, (255,0,0), -1)
                    x4,y4 = c[1].astype('int32')
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 5)
            cv2.line(frame, (x2,y2), (x4,y4), (0,255,0), 5)
            cv2.line(frame, (x4,y4), (x3,y3), (0,255,0), 5)
            cv2.line(frame, (x3,y3), (x1,y1), (0,255,0), 5)

            pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
            pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(frame,M,(500,500))
            cv2.imshow('dst', dst)
            #if press 's' key, save image
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite('photo_test_image.jpg', dst)
                print('image saved')
                break
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

dst_final = cv2.imread('photo_test_image.jpg')

lineWidth = 7
lineMinWidth = 55
gray_scale=cv2.cvtColor(dst_final,cv2.COLOR_BGR2GRAY)
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
#cv2.imshow("Image horizontal lines", img_bin_h)
img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1v)  # bridge small gap in vert lines
img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)# kep ony vert lines by eroding everything else in vert direction
#cv2.imshow("Image vertical lines", img_bin_v)
def fix(img):
    img[img>127]=255
    img[img<127]=0
    return img

img_bin_final = fix(fix(img_bin_h)|fix(img_bin_v))
finalKernel = np.ones((5,5), np.uint8)
img_bin_final=cv2.dilate(img_bin_final,finalKernel,iterations=1)
#img_bin_final=cv2.erode(img_bin_final,finalKernel,iterations=1)
coordinates_rectangles=[]
ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
count_rect = 0
for x,y,w,h,area in stats[2:]:
    count_rect += 1
    cv2.rectangle(dst_final,(x,y),(x+w,y+h),(0,255,0),2)
    coordinates_rectangles.append([y, y+h,x, x+w])
    rectangle = img_bin[y:y+h,x:x+w]
    count_pixels_blk = 0
    count_pixels_wht = 0
    count_pixels = 0
    #read pixels in rectangle
    for i in rectangle:
        for j in i:
            count_pixels += 1
            if j == 0:
                count_pixels_blk += 1
            else:
                count_pixels_wht += 1
    percentage_blk = (count_pixels_blk/count_pixels)*100
    percentage_wht = (count_pixels_wht/count_pixels)*100
    if percentage_blk > 10 and percentage_wht > 30:
         cv2.circle(dst_final, (x, y), 5, (255,0,0), -1)
    if percentage_blk > 70:
        cv2.circle(dst_final, (x, y), 5, (0,0,255), -1)

cv2.imwrite("photo_final_dst.jpg", dst_final)
