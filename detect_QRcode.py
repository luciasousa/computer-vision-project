#detect and decode QR code with OpenCV
import cv2

#load image
image = cv2.imread("qr.jpg")    

#load QR code detector
detector = cv2.QRCodeDetector()

decoded_info = []
points = []
straight_qrcode = []

#detect and decode more than one QR code
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
'''
#with video
#load QR code detector
detector = cv2.QRCodeDetector()

#open video
cap = cv2.VideoCapture(0)

while True:
    #read frame
    _, frame = cap.read()

    #detect and decode
    data, bbox, straight_qrcode = detector.detectAndDecode(frame)

    #display data
    if bbox is not None:
        for i in range(len(bbox)):
            #draw all lines
            cv2.line(frame, tuple(bbox[i][0]), tuple(bbox[(i+1) % len(bbox)][0]), color=(255,0,0), thickness=2)
        if data:
            print("[+] QR Code detected, data:", data)

    #display the result
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''