#programa que tira foto ao teste e depois faz o processamento
import argparse
import imutils
import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np

'''
def init_matrix(matrix):
    #first row of matrix is None,a,b,c,d,None,a,b,c,d,None,a,b,c,d
    matrix[0][0] = None
    matrix[0][1] = 'a'
    matrix[0][2] = 'b'
    matrix[0][3] = 'c'
    matrix[0][4] = 'd'
    matrix[0][5] = None
    matrix[0][6] = 'a'
    matrix[0][7] = 'b'
    matrix[0][8] = 'c'
    matrix[0][9] = 'd'
    matrix[0][10] = None
    matrix[0][11] = 'a'
    matrix[0][12] = 'b'
    matrix[0][13] = 'c'
    matrix[0][14] = 'd'

    #first column of matrix is None,1,2,3,4,5,6,7,8,9,10,11,12,13,14
    matrix[1][0] = 1
    matrix[2][0] = 2
    matrix[3][0] = 3
    matrix[4][0] = 4
    matrix[5][0] = 5
    matrix[6][0] = 6
    matrix[7][0] = 7
    matrix[8][0] = 8
    matrix[9][0] = 9
    matrix[10][0] = 10
    matrix[11][0] = 11
    matrix[12][0] = 12
    matrix[13][0] = 13
    matrix[14][0] = 14

    #sixt column of matrix is None,15,16,17,18,19,20,21,22,23,24,25,26,27,28
    matrix[1][5] = 15
    matrix[2][5] = 16
    matrix[3][5] = 17
    matrix[4][5] = 18
    matrix[5][5] = 19
    matrix[6][5] = 20
    matrix[7][5] = 21
    matrix[8][5] = 22
    matrix[9][5] = 23
    matrix[10][5] = 24
    matrix[11][5] = 25
    matrix[12][5] = 26
    matrix[13][5] = 27
    matrix[14][5] = 28

    #eleven column of matrix is None,29,30,31,32,33,34,35,36,37,38,39,40,41,42
    matrix[1][10] = 29
    matrix[2][10] = 30
    matrix[3][10] = 31
    matrix[4][10] = 32
    matrix[5][10] = 33
    matrix[6][10] = 34
    matrix[7][10] = 35
    matrix[8][10] = 36
    matrix[9][10] = 37
    matrix[10][10] = 38
    matrix[11][10] = 39
    matrix[12][10] = 40
    matrix[13][10] = 41
    matrix[14][10] = 42

    return matrix
'''

def map_coordinates_x(coordinates_rectangles):

    mapx = 0
    mapx_list = []
    x0=coordinates_rectangles[0][0]
    x_velho = []
    for i in coordinates_rectangles:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3] 
        tempx = x-x0
        #print("w = ", w)
        x_velho_aux=[x,mapx]
        if tempx > w and w > 15:
            x0 = x
            mapx_list.append(mapx)
            mapx += 1
            x_velho_aux=[x,mapx]
        x_velho.append(x_velho_aux)
   # print(x_velho)
    return mapx_list,x_velho


def map_coordinates_y(coordinates_rectangles):
    #print(coordinates_rectangles)
    mapy = 0
    mapy_list = []
    y0=coordinates_rectangles[0][1]
    y_velho = []
    for i in coordinates_rectangles:
        x= i[0]
        y=i[1]
        w=i[2]
        h=i[3] 
        y_velho_aux= [y,mapy]
        #print("x, y, w, h= ",x, y, w, h)
        #print("h = ", h)
        tempy = y-y0
        if tempy > h:
            y0 = y
            mapy_list.append(mapy)
            mapy += 1
            y_velho_aux=[y,mapy]
        y_velho.append(y_velho_aux)
   # print(y_velho)
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
ss_count = 0
percentage_blk=0
percentage_wht=0
list_x = [] #list with x coordinates
list_y = [] #list with y coordinates
#matrix with 15 rows and 15 columns
matrix = [[0 for x in range(15)] for y in range(15)]

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
x_velho=[]
y_velho=[]
coordinates_rectangles.sort(key=lambda x: x[0])

new_x,x_velho = map_coordinates_x(coordinates_rectangles)
coordinates_rectangles.sort(key=lambda x: x[1])
#print(coordinates_rectangles)
new_y , y_velho= map_coordinates_y(coordinates_rectangles)

#print(coordinates_rectangles)

print(new_x)
print(new_y)

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
    
    #mapear x e y para valores das colunas e linhas

    percentage_blk = i[4]
    percentage_wht = i[5]
    if percentage_blk > 15 and percentage_wht > 30: #has an x
        cv2.circle(dst_final, (x, y), 5, (255,0,0), -1)
        #print("x, y: ", x_novo, y_novo)
        matrix[y_novo][x_novo] = 1
    if percentage_blk > 80: 
        cv2.circle(dst_final, (x, y), 5, (0,0,255), -1) 

#init_matrix(matrix)
for row in matrix:
    print(row)
    


cv2.imwrite("photo_final_dst.jpg", dst_final)


#remove first line from matrix
matrix.pop(0)
#remove first column from matrix
for row in matrix:
    row.pop(0)
    row.pop(4)
    row.pop(8)


print("matrix pop")
for row in matrix:
    print(row)

#create new matrix with 14 rows and 12 columns
matrix_14_12 = [[0 for x in range(12)] for y in range(14)]

#copy values from matrix to matrix_14_12
for i in range(14):
    for j in range(12):
        matrix_14_12[i][j] = matrix[i][j]

print("matrix_14_12 -----------------------------")
for row in matrix_14_12:
    print(row)

col_min = 0
col_max = 4

count_res = 0

array_answers = [0 for x in range(42)]

for i in range(14):
    count_res = 0
    for j in range(col_min, col_max):
        if matrix_14_12[i][j] == 1:
            count_res = 1
            if j == 0:
                array_answers[i] = 'a'
            if j == 1: 
                array_answers[i] = 'b'
            if j == 2:
                array_answers[i] = 'c'
            if j == 3:
                array_answers[i] = 'd'
        elif count_res == 0:
            array_answers[i] = '-'

col_min += 4
col_max += 4

for i in range(14):
    i_aux = i
    count_res = 0
    i_aux += 14
    for j in range(col_min, col_max):
        if matrix_14_12[i][j] == 1:
            count_res = 1
            if j == 4:
                array_answers[i_aux] = 'a'
            if j == 5: 
                array_answers[i_aux] = 'b'
            if j == 6:
                array_answers[i_aux] = 'c'
            if j == 7:
                array_answers[i_aux] = 'd'
        elif count_res == 0:
            array_answers[i_aux] = '-'

col_min += 4
col_max += 4

for i in range(14):
    i_aux = i
    count_res = 0
    i_aux += 28
    for j in range(col_min, col_max):
        if matrix_14_12[i][j] == 1:
            count_res = 1
            if j == 8:
                array_answers[i_aux] = 'a'
            if j == 9: 
                array_answers[i_aux] = 'b'
            if j == 10:
                array_answers[i_aux] = 'c'
            if j == 11:
                array_answers[i_aux] = 'd'
        elif count_res == 0:
            array_answers[i_aux] = '-'

print(array_answers)
'''
#create matrix with 42 rows and 4 columns
matrix_42_4 = [[0 for x in range(4)] for y in range(42)]

col_min = 0
col_max = 3

range_i = 14

lines = 0
lines_aux = 0

for j in range(col_min, col_max):
    for i in range(14):
        matrix_42_4[lines][j//3] = matrix_14_12[i][j]
        lines = i + lines_aux
    
    lines_aux += 14
    col_min = col_min + 4
    col_max = col_max + 4
    

#copy values from matrix_14_12 to matrix_42_4
for i in range(14):
    for j in range(col_min, col_max):
        matrix_42_4[lines][j//3] =0# matrix_14_12[i][j]
        lines += 1

    
    col_min = col_min + 4
    col_max = col_max + 4
    print ("lines, cols: ", lines, col_min, col_max)


print("matrix_42_4 -----------------------------")
for row in matrix_42_4:
    print(row)
'''