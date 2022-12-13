#detetc Aruco markers with webcam
import argparse
import imutils
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

cam = cv2.VideoCapture(0)
flag = 0
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
        """
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
        dst[:,:,0] = cv2.equalizeHist(dst[:,:,0])
        dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
        """

        #contrast stretching
        """
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
        dst[:,:,0] = cv2.normalize(dst[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
        """

        ## binarising image
        #adaptative thresholding

        lineWidth = 7
        lineMinWidth = 55
        #kernal1 = np.ones((lineWidth,lineWidth), np.uint8)
        gray_scale=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        #se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        #bg=cv2.morphologyEx(gray_scale, cv2.MORPH_DILATE, se)
        #out_gray=cv2.divide(gray_scale, bg, scale=255)
        #img_bin=cv2.threshold(gray_scale, 0, 255, cv2.THRESH_OTSU )[1] 

        #dilate image
        #gray_scale_dil = cv2.erode(gray_scale, kernal1, iterations=1)
        th1,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
        #img_bin = cv2.adaptiveThreshold(gray_scale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

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
        cv2.imshow("Image bin final dilate" , img_bin_final)
        img_bin_final=cv2.erode(img_bin_final,finalKernel,iterations=1)
        cv2.imshow("Image bin final erode", img_bin_final)
        coordinates_rectangles=[]
        ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        count_rect = 0
        for x,y,w,h,area in stats[2:]:
            count_rect += 1
            cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
            coordinates_rectangles.append([y, y+h,x, x+w])
        #print(coordinates_rectangles)
        print("Number of rectangles = " + str(count_rect))
        if count_rect == 224: #210
            #sort rectangles by x and y coordinates
            #coordinates_rectangles.sort(key=lambda x: x[0])
            #coordinates_rectangles.sort(key=lambda x: x[2])
            #print(coordinates_rectangles)
            #iterate over all rectangles by coordinates
            for x,y,w,h,area in stats[2:]:
                rectangle = img_bin[y:y+h,x:x+w]
                #contours
                contours, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #cv2.drawContours(rectangle, contours, -1, (0,255,0), 3)
                flag=1
                if (len(contours) == 4): #if has x
                    print("rectangle 4 = " + str(x) + ' ' + str(y))
                    print("Number of contours = " + str(len(contours)))
                    #draw small circle in the center of the rectangle
                    cv2.circle(dst, (x, y), 5, (255,0,0), -1)

                    cv2.imshow("contours", rectangle)
                    cv2.waitKey(0)
                
                if (len(contours) == 0): #if the rectangle is black
                    print("rectangle 0 = " + str(x) + ' ' + str(y))
                    print("Number of contours = " + str(len(contours)))
                    #draw small circle in the center of the rectangle
                    cv2.circle(dst, (x, y), 5, (0,0,255), -1)
                    cv2.imshow("contours", rectangle)
                    cv2.waitKey(0)
                

                
                
            '''
            #iterate over all rectangles
            for x,y,w,h,area in stats[2:]:
                rectangle = img_bin[y:y+h,x:x+w]
                #contours
                contours, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rectangle, contours, -1, (0,255,0), 3)
                flag=1
                if (len(contours) == 4):
                    print("Number of rectangle = " + str(count_rect))
                    print("Number of contours = " + str(len(contours)))
                    #draw small circle in the center of the rectangle
                    cv2.circle(dst, (int(w/2),int(h/2)), 5, (255,0,0), -1)
                if (len(contours) == 0):
                    print("Number of rectangle = " + str(count_rect))
                    print("Number of contours = " + str(len(contours)))
                    #draw small circle in the center of the rectangle
                    cv2.circle(dst, (int(w/2),int(h/2)), 5, (0,0,255), -1)
            '''
            
        
            
    cv2.imshow("Image", frame)
    cv2.imshow("Perspective correction with adaptative thresholding", img_bin)
    cv2.imshow("Dest", dst)
    #save image
    cv2.imwrite("dst.jpg", dst)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or flag==1:
        break

cv2.destroyAllWindows()
cam.release()