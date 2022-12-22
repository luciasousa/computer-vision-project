#programa que tira foto ao teste e depois faz o processamento
import argparse
import imutils
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

NUMBER_OF_QUESTIONS = 37
NUMBER_OF_QUESTIONS_VF = 12
NUMBER_OF_LINES = 14
NUMBER_OF_COLUMNS = 3
NUMBER_OF_COLUMNS_VF = 1

NUMBER_OF_COLUMNS_TOTAL = NUMBER_OF_COLUMNS*4+NUMBER_OF_COLUMNS
NUMBER_OF_COLUMNS_TOTAL_VF = NUMBER_OF_COLUMNS_VF*2+NUMBER_OF_COLUMNS_VF

max_limit = (NUMBER_OF_COLUMNS_TOTAL+NUMBER_OF_COLUMNS_TOTAL_VF) * NUMBER_OF_LINES + 10
min_limit = (NUMBER_OF_COLUMNS_TOTAL+NUMBER_OF_COLUMNS_TOTAL_VF) * NUMBER_OF_LINES - 10

def map_coordinates_x(coordinates_rectangles):
    mapx = 0
    mapx_list = []
    x0=coordinates_rectangles[0][0]
    x_velho = []
    for i in coordinates_rectangles:
        x=i[0]
        w=i[2]
        tempx = x-x0
        x_velho_aux=[x,mapx]
        if tempx > w and w > w/2:
            x0 = x
            mapx_list.append(mapx)
            mapx += 1
            x_velho_aux=[x,mapx]
        x_velho.append(x_velho_aux)
    return mapx_list,x_velho


def map_coordinates_y(coordinates_rectangles):
    mapy = 0
    mapy_list = []
    y0=coordinates_rectangles[0][1]
    y_velho = []
    for i in coordinates_rectangles:
        y=i[1]
        h=i[3] 
        y_velho_aux= [y,mapy]
        tempy = y-y0
        if tempy > h and h > h/2:
            y0 = y
            mapy_list.append(mapy)
            mapy += 1
            y_velho_aux=[y,mapy]
        y_velho.append(y_velho_aux)
    return mapy_list, y_velho


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
        if length == 6: #meter 4 se nao tiver perguntas v/F
            for i in range(len(corners)):
                c = corners[i][0]# c type = numpy.ndarray
                #cv2.line(frame, tuple(c[0].astype('int32')), tuple(c[1].astype('int32')), (255,0,255), 5)
                #cv2.line(frame, tuple(c[1].astype('int32')), tuple(c[2].astype('int32')), (255,0,255), 5)
                #cv2.line(frame, tuple(c[2].astype('int32')), tuple(c[3].astype('int32')), (255,0,255), 5)
                #cv2.line(frame, tuple(c[3].astype('int32')), tuple(c[0].astype('int32')), (255,0,255), 5)
                if ids[i] == 1:
                    #cv2.circle(frame, tuple(c[3].astype('int32')), 5, (255,0,0), -1)
                    x1,y1 = c[3].astype('int32')
                if ids[i] == 2:
                    #cv2.circle(frame, tuple(c[2].astype('int32')), 5, (255,0,0), -1)
                    x2,y2 = c[2].astype('int32')
                if ids[i] == 3:
                    #cv2.circle(frame, tuple(c[0].astype('int32')), 5, (255,0,0), -1)
                    x3,y3 = c[0].astype('int32')
                if ids[i] == 9: 
                    #cv2.circle(frame, tuple(c[2].astype('int32')), 5, (255,0,0), -1)
                    x9,y9 = c[2].astype('int32')
                if ids[i] == 5:
                    #cv2.circle(frame, tuple(c[1].astype('int32')), 5, (255,0,0), -1)
                    x5,y5 = c[1].astype('int32')
                if ids[i] == 8:
                    #cv2.circle(frame, tuple(c[1].astype('int32')), 5, (255,0,0), -1)
                    x8,y8 = c[1].astype('int32')
            

            cv2.line(frame, (x1,y1), (x9,y9), (0,255,0), 5)
            cv2.line(frame, (x1,y1), (x3,y3), (0,255,0), 5)
            cv2.line(frame, (x3,y3), (x8,y8), (0,255,0), 5)
            cv2.line(frame, (x9,y9), (x8,y8), (0,255,0), 5)

            #perspective correction
            pts1 = np.float32([[x1,y1],[x9,y9],[x3,y3],[x8,y8]])
            #pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x5,y5]])
            pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(frame,M,(500,500))
            cv2.imshow('dst', dst)

            dst_final = dst

            lineWidth = 7
            lineMinWidth = 55
            gray_scale=cv2.cvtColor(dst_final,cv2.COLOR_BGR2GRAY)
            th1,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
            kernal1 = np.ones((lineWidth,lineWidth), np.uint8)
            kernal1h = np.ones((1,lineWidth), np.uint8)
            kernal1v = np.ones((lineWidth,1), np.uint8)

            kernal6 = np.ones((lineMinWidth,lineMinWidth), np.uint8)
            kernal6h = np.ones((1,lineMinWidth), np.uint8)
            kernal6v = np.ones((lineMinWidth,1), np.uint8)

            img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1h) # bridge small gap in horizonntal lines
            img_bin_h = cv2.morphologyEx(img_bin_h, cv2.MORPH_OPEN, kernal6h) # kep ony horiz lines by eroding everything else in hor direction
            img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_CLOSE, kernal1v)  # bridge small gap in vert lines
            img_bin_v = cv2.morphologyEx(img_bin_v, cv2.MORPH_OPEN, kernal6v)# kep ony vert lines by eroding everything else in vert direction
            def fix(img):
                img[img>127]=255
                img[img<127]=0
                return img

            img_bin_final = fix(fix(img_bin_h)|fix(img_bin_v))
            finalKernel = np.ones((5,5), np.uint8)
            img_bin_final=cv2.dilate(img_bin_final,finalKernel,iterations=1)
            coordinates_rectangles=[]
            ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
            count_rect = 0
            ss_count = 0
            percentage_blk=0
            percentage_wht=0
            list_x = [] #list with x coordinates
            list_y = [] #list with y coordinates
            matrix_questions = [[0 for x in range(NUMBER_OF_COLUMNS_TOTAL+NUMBER_OF_COLUMNS_TOTAL_VF)] for y in range(NUMBER_OF_LINES+1)]

            for x,y,w,h,area in stats[2:]:
                ss_count += 1
                count_rect += 1
                cv2.rectangle(dst_final,(x,y),(x+w,y+h),(0,255,0),2)
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
                coordinates_rectangles.append([x, y, w, h, percentage_blk, percentage_wht])
           
            #sort coordinates by x and y
            if count_rect > min_limit or count_rect < max_limit: #225
                x_velho=[]
                y_velho=[]
                coordinates_rectangles.sort(key=lambda x: x[0])
                new_x,x_velho = map_coordinates_x(coordinates_rectangles)
                coordinates_rectangles.sort(key=lambda x: x[1])
                new_y , y_velho= map_coordinates_y(coordinates_rectangles)
                x_novo = 0
                y_novo = 0

                for i in coordinates_rectangles:
                    x = i[0]
                    y = i[1]
                    for x_v in x_velho:
                        if x_v[0] == x:
                            x_novo = x_v[1]
                    for y_v in y_velho:
                        if y_v[0] == y:
                            y_novo = y_v[1]
                    percentage_blk = i[4]
                    percentage_wht = i[5]
                    if percentage_blk > 15 and percentage_wht > 30: #has an x
                        cv2.circle(dst_final, (x, y), 5, (255,0,0), -1)
                        matrix_questions[y_novo][x_novo] = 1
                    if percentage_blk > 80: 
                        cv2.circle(dst_final, (x, y), 5, (0,0,255), -1) 
                
                cv2.circle(frame, (10,15), 10, (0,255,0), -1)
                cv2.imwrite("./images/live_final_dst.jpg",dst_final)
                print("correction done")
            
                #remove first line from matrix
                matrix_questions.pop(0)

                #remove first column from matrix
                for row in matrix_questions:
                    r = 0
                    row.pop(r)
                    for i in range(NUMBER_OF_COLUMNS-1):
                        r +=4
                        row.pop(r)
                    r +=4
                    for i in range(NUMBER_OF_COLUMNS_VF):
                        row.pop(r)
                        r +=2

                #create new matrix with 14 rows and 12 columns adicionar + NUMBER_OF_COLUMNS_VF*2
                matrix_questions_final = [[0 for x in range(NUMBER_OF_COLUMNS*4 + NUMBER_OF_COLUMNS_VF*2)] for y in range(NUMBER_OF_LINES)]

                #copy values from matrix to matrix_14_12 + NUMBER_OF_COLUMNS_VF*2
                for i in range(NUMBER_OF_LINES):
                    for j in range(NUMBER_OF_COLUMNS*4 + NUMBER_OF_COLUMNS_VF*2):
                        matrix_questions_final[i][j] = matrix_questions[i][j]

                col_min = 0
                col_max = 4

                count_res = 0

                array_answers = [0 for x in range(NUMBER_OF_QUESTIONS)]
                array_answers_vf = [0 for x in range(NUMBER_OF_QUESTIONS_VF)]
                i_aux = 0
                for i in range(NUMBER_OF_COLUMNS):
                    for i in range(NUMBER_OF_LINES):
                        count_res = 0
                        for j in range(col_min, col_max):
                            if matrix_questions_final[i][j] == 1:
                                count_res = 1
                                if j%4==0:
                                    array_answers[i_aux] = 'a'
                                if (j-1)%4==0:
                                    array_answers[i_aux] = 'b'
                                if (j-2)%4==0:
                                    array_answers[i_aux] = 'c'
                                if (j-3)%4==0:
                                    array_answers[i_aux] = 'd'
                            elif count_res == 0:
                                array_answers[i_aux] = '-'
                        i_aux += 1
                        if i_aux == NUMBER_OF_QUESTIONS:
                            break
                    col_min += 4
                    col_max += 4

                i_aux = 0
                col_max = col_max-2

                for k in range(NUMBER_OF_COLUMNS_VF):
                    for i in range(NUMBER_OF_LINES):
                        count_res = 0
                        for j in range(col_min, col_max):
                            if matrix_questions_final[i][j] == 1:
                                count_res = 1
                                if j%2==0:
                                    array_answers_vf[i_aux] = 'v'
                                else:
                                    array_answers_vf[i_aux] = 'f'
                            elif count_res == 0:
                                array_answers_vf[i_aux] = '-'
                        i_aux += 1
                        if i_aux == NUMBER_OF_QUESTIONS_VF:
                            break
                    col_min += 2
                    col_max += 2

                print(array_answers)
                print(array_answers_vf)

            else:
                cv2.circle(frame, (10,15), 10, (255,0,0), -1)

            cv2.imshow("correction", dst_final)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
