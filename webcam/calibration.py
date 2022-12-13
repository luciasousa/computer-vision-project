
import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def  FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)
    # If found, display image with corners
    if ret == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('video',img)
        cv2.waitKey(500)

    return ret, corners

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w*board_h,3), np.float32)
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# Video

capture = cv2.VideoCapture(0)
window_size = (5 ,5)
zero_zone = (-1,-1)
criteria = None
j = 0
while(True):
    # Capture frame-by-frame
    det, frame = capture.read()
    cv2.imshow('video',frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = FindAndDisplayChessboard(frame)
    if ret == True:
        #if key 's' is pressed, save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            corners = cv2.cornerSubPix(gray, corners, winSize=window_size, zeroZone=zero_zone, criteria=criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.imwrite('images_camera_calibration/image%d.jpg' % j, frame)
            print('image%d.jpg saved' % j)
            j += 1
            print(j)
    if j == 10:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('camera_params.npz', intrinsics=intrinsics, distortion=distortion, rvecs=rvecs, tvecs=tvecs)
