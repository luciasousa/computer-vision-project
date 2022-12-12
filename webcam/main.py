#detetc Aruco markers with webcam
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
    
    #if detected Aruco markers
    if ids is not None:
        #print("ids " + str(ids))
        for i in range(len(corners)):
            c = corners[i][0]# c type = numpy.ndarray
            cv2.line(frame, tuple(c[0].astype('int32')), tuple(c[1].astype('int32')), (255,0,255), 5)
            cv2.line(frame, tuple(c[1].astype('int32')), tuple(c[2].astype('int32')), (255,0,255), 5)
            cv2.line(frame, tuple(c[2].astype('int32')), tuple(c[3].astype('int32')), (255,0,255), 5)
            cv2.line(frame, tuple(c[3].astype('int32')), tuple(c[0].astype('int32')), (255,0,255), 5)
            #cv2.putText(frame, str(c[0]), tuple(c[0].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            #cv2.putText(frame, str(c[1]), tuple(c[1].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            #cv2.putText(frame, str(c[2]), tuple(c[2].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            #cv2.putText(frame, str(c[3]), tuple(c[3].astype('int32')), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            #draw circle in left down corner of the marker with the id number equal to 1
            if ids[i] == 1:
                cv2.circle(frame, tuple(c[3].astype('int32')), 5, (255,0,0), -1)
                x1,y1 = c[3].astype('int32')
                #print("x1,y1: ", x1,y1)
            #draw circle in right down corner of the marker with the id number equal to 2
            if ids[i] == 2:
                cv2.circle(frame, tuple(c[2].astype('int32')), 5, (255,0,0), -1)
                x2,y2 = c[2].astype('int32')
                #print("x2,y2: ", x2,y2)
            #draw circle in left up corner of the marker with the id number equal to 3
            if ids[i] == 3:
                cv2.circle(frame, tuple(c[0].astype('int32')), 5, (255,0,0), -1)
                x3,y3 = c[0].astype('int32')
                #print("x3,y3: ", x3,y3)
            #draw circle in right up corner of the marker with the id number equal to 4
            if ids[i] == 5:
                cv2.circle(frame, tuple(c[1].astype('int32')), 5, (255,0,0), -1)
                x4,y4 = c[1].astype('int32')
                #print("x4,y4: ", x4,y4)

        #draw rectangle around the detected x1,y1,x2,y2,x3,y3,x4,y4
        cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 5)
        cv2.line(frame, (x2,y2), (x4,y4), (0,255,0), 5)
        cv2.line(frame, (x4,y4), (x3,y3), (0,255,0), 5)
        cv2.line(frame, (x3,y3), (x1,y1), (0,255,0), 5)

        #perspective correction
        pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(frame,M,(500,500))

        #histogram equalization
        '''
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
        dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
        dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
        '''

        #contrast stretching
        '''
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
        dst[:,:,0] = cv2.normalize(dst[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
        '''

        ## binarising image
        #adaptative thresholding
        gray_scale=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        th1,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
        #img_bin = cv2.adaptiveThreshold(gray_scale,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

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
        def fix(img):
            img[img>127]=255
            img[img<127]=0
            return img


        img_bin_final = fix(fix(img_bin_h)|fix(img_bin_v))
        finalKernel = np.ones((5,5), np.uint8)
        img_bin_final=cv2.dilate(img_bin_final,finalKernel,iterations=1)

        ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

        

        for x,y,w,h,area in stats[2:]:
           # if x>=x1 and x<=x2 and y>=y1 and y<=y3:
            cv2.imshow("Image", frame)
            cv2.imshow("Perspective correction with adaptative thresholding", img_bin)
            cv2.rectangle(img_bin,(x,y),(x+w,y+h),(0,255,0),2)
            rectangle = img_bin[y:y+h,x:x+w]
            #contours
            contours, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rectangle, contours, -1, (0,255,0), 3)
            print("Number of contours = " + str(len(contours)))
            #show contours
            
            cv2.imshow("Image contours", rectangle)
            cv2.waitKey(0)  
            '''
                
                plt.figure(figsize=(1,1))
                plt.imshow(rectangle)
                plt.show()'''
        

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
            