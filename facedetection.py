import cv2
import os
alg = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)

flag=0
while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
        flag=1
        cv2.imshow("FaceDetection",img)
    if flag==1:
        print("Person detected")
    else:
        print("Person not detected")
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
