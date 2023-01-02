#program that finds the bigger area (grid), save image by press key 's' and process image to find 'x' and get the sequence of answers

import argparse
import imutils
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

#define variables
NUMBER_OF_QUESTIONS = 42
NUMBER_OF_QUESTIONS_VF = 14
NUMBER_OF_LINES = 14
NUMBER_OF_COLUMNS = 3
NUMBER_OF_COLUMNS_VF = 1
NUMBER_OF_COLUMNS_TOTAL = NUMBER_OF_COLUMNS*4+NUMBER_OF_COLUMNS
NUMBER_OF_COLUMNS_TOTAL_VF = NUMBER_OF_COLUMNS_VF*2+NUMBER_OF_COLUMNS_VF

#map x coordinates of boxes in the grid to the matrix lines
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

#map y coordinates of boxes in the grid to the matrix columns
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

#start capture
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    th1 = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    contours, hierarchy = cv2.findContours(image=th1, mode = cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    max_area = 0
    c=0
    #find the biggest area
    for i in contours:
        area = cv2.contourArea(i)
        if area>1000:
            if area> max_area:
                max_area=area
                best_count=i
                cv2.drawContours(frame,contours,c,(0,255,0),3)
    mask = np.zeros((gray.shape),np.uint8)
    cv2.drawContours(mask, [best_count],0,255,-1)
    cv2.drawContours(mask,[best_count],0,0,2)

    #perspective correction with contours
    peri = cv2.arcLength(best_count, True)
    approx = cv2.approxPolyDP(best_count, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 2)
        pts1 = np.float32([screenCnt[0],screenCnt[1],screenCnt[2],screenCnt[3]])
        pts2 = np.float32([[0,0],[0,500],[500,500],[500,0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(frame,M,(500,500))
        cv2.imshow('dst', dst)
        cv2.imshow('frame', frame)

    #if press 's' key, save image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('./images/photo_test_image_without.jpg', dst)
        print('image saved')
        break
    cv2.imshow('frame', frame)

    #if press 'q' key, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#read image saved and process
dst_final = cv2.imread('./images/photo_test_image_without.jpg')

#obtain the vertical and horizontal lines of the grid
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

#find the coordinates of the boxes
coordinates_rectangles=[]
ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
count_rect = 0
ss_count = 0
percentage_blk=0
percentage_wht=0
matrix_questions = [[0 for x in range(NUMBER_OF_COLUMNS_TOTAL+NUMBER_OF_COLUMNS_TOTAL_VF)] for y in range(NUMBER_OF_LINES+1)]

#read each box and get the percentage of black and white pixels
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

x_velho=[]
y_velho=[]
#sort the x coordinates of the boxes
coordinates_rectangles.sort(key=lambda x: x[0])
#map the x coordinates of the boxes to matrix lines
new_x,x_velho = map_coordinates_x(coordinates_rectangles)
#sort the y coordinates of the boxes
coordinates_rectangles.sort(key=lambda x: x[1])
#map the y coordinates of the boxes to matrix columns
new_y , y_velho= map_coordinates_y(coordinates_rectangles)
x_novo = 0
y_novo = 0

#read the boxes and check if they have an x
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
    #if the percentage of black pixels is greater than 15% and the percentage of white pixels is greater than 30% then it has an x
    if percentage_blk > 15 and percentage_wht > 30:
        cv2.circle(dst_final, (x, y), 5, (255,0,0), -1)
        #put 1 in the matrix
        if y_novo < NUMBER_OF_LINES and x_novo < NUMBER_OF_COLUMNS_TOTAL:
            matrix_questions[y_novo][x_novo] = 1
    #if the percentage of black pixels is greater than 80% then it is filled
    if percentage_blk > 80: 
        cv2.circle(dst_final, (x, y), 5, (0,0,255), -1) 

#save image with x marked with blue circles and filled boxes with red circles
cv2.imwrite("./images/photo_final_dst_without.jpg", dst_final)

#remove first line from matrix
matrix_questions.pop(0)

#remove columns from matrix that are not needed (columns with the numbers)
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

#create new matrix without the first line and the columns with numbers of questions
matrix_questions_final = [[0 for x in range(NUMBER_OF_COLUMNS*4 + NUMBER_OF_COLUMNS_VF*2)] for y in range(NUMBER_OF_LINES)]

#copy values to new matrix
for i in range(NUMBER_OF_LINES):
    for j in range(NUMBER_OF_COLUMNS*4 + NUMBER_OF_COLUMNS_VF*2):
        matrix_questions_final[i][j] = matrix_questions[i][j]

col_min = 0
col_max = 4
count_res = 0
#find the answer of each question
array_answers = [0 for x in range(NUMBER_OF_QUESTIONS)]
array_answers_vf = [0 for x in range(NUMBER_OF_QUESTIONS_VF)]
i_aux = 0
for k in range(NUMBER_OF_COLUMNS):
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
