#detect and decode QR code with OpenCV
import cv2

#load image
image = cv2.imread("../Automatic_correction_image/example3.png")    

#load QR code detector
detector = cv2.QRCodeDetector()

retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(image)

#get qr code positions
'''
if retval == True:
    for i in range(len(points)):
        #get points
        points = points[i]
        #get points
        p1 = (points[0][0].astype('int32'), points[0][1].astype('int32'))
        p2 = (points[1][0].astype('int32'), points[1][1].astype('int32'))
        p3 = (points[2][0].astype('int32'), points[2][1].astype('int32'))
        p4 = (points[3][0].astype('int32'), points[3][1].astype('int32'))
        cv2.line(image, p1, p2, (255,0,0), 2)
        cv2.line(image, p2, p3, (255,0,0), 2)
        cv2.line(image, p3, p4, (255,0,0), 2)
        cv2.line(image, p4, p1, (255,0,0), 2)
        cv2.putText(image, str(p1), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(image, str(p2), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(image, str(p3), p3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(image, str(p4), p4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
'''
#check if QR code is detected and draw polyline around it and print data

if len(decoded_info) > 0:
    for i in range(len(decoded_info)):
        print("QR Code detected")
        print("Decoded Data : {}".format(decoded_info[i]))
        #polyline around QR code
        n = len(points[i])  
        for j in range(n):
            p1 = (points[i][j][0].astype('int32'), points[i][j][1].astype('int32'))
            p2 = (points[i][(j+1) % n][0].astype('int32'), points[i][(j+1) % n][1].astype('int32'))
            p3 = (points[i][(j+2) % n][0].astype('int32'), points[i][(j+2) % n][1].astype('int32'))
            p4 = (points[i][(j+3) % n][0].astype('int32'), points[i][(j+3) % n][1].astype('int32'))
            cv2.line(image, tuple(points[i][j].astype('int32')), tuple(points[i][(j+1) % n].astype('int32')), (255,0,0), 5)
            cv2.putText(image, str(p1), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(image, str(p2), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(image, str(p3), p3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(image, str(p4), p4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


#display image
cv2.imshow("QR Code Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()