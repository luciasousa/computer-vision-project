#detect and decode QR code with OpenCV
import cv2

#load image
image = cv2.imread("../Automatic_correction_image/example1.png")    

#load QR code detector
detector = cv2.QRCodeDetector()

decoded_info = []
points = []
straight_qrcode = []

retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(image)

#check if QR code is detected and draw polyline around it and print data
if len(decoded_info) > 0:
    for i in range(len(decoded_info)):
        print("QR Code detected")
        print("Decoded Data : {}".format(decoded_info[i]))
        #polyline around QR code
        n = len(points[i])  
        for j in range(n):
            cv2.line(image, tuple(points[i][j].astype('int32')), tuple(points[i][(j+1) % n].astype('int32')), (255,0,0), 5)

#display image
cv2.imshow("QR Code Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

çççççç