import cv2
import numpy as np 
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

cap=cv2.VideoCapture(0)

while True:
    ret, color_img=cap.read()
    gray_img=cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray_img)
    for(x,y,w,h) in faces:
        cv2.rectangle(color_img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(color_img,'FACE',(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        face_gray=gray_img[y:y+h,x:x+w]
        face_color=color_img[y:y+h,x:x+w]


        eyes=eye_cascade.detectMultiScale(face_gray)  
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(face_color,'EYES',(ex,ey),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    cv2.imshow('img',color_img)
    k=cv2.waitKey(30) & 0xff
    if k==113:
        break
cap.release()
cv2.destroyAllWindows()

