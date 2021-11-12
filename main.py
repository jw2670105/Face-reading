import os
import cv2
import sys
from tkinter import filedialog
from PIL import Image
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
names=[]

path='C:/Users/95198/Desktop/face'
imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
for imagePath in imagePaths:
    name = str(os.path.split(imagePath)[1].split('.',2)[1])
    names.append(name)


a=0
while (a!=1 and a!=2):
    a=input("Type 1 to turn on picture scan mode. Type 2 to turn on camera mode ")
    a=int(a)
    if (a==1):
        file_path = filedialog.askopenfilename()
        while True:
            img = cv2.imread(file_path, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))

            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                ids, score = recognizer.predict(gray[y:y+h, x:x+w])
                if (score>30):
                    cv2.putText(img, 'stranger', (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0),1 )
                else:
                    cv2.putText(img, str(names[ids-1]), (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0),1 )
            cv2.imshow('picture', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()


    if (a==2):
        video_capture = cv2.VideoCapture(0)
        while True:
            retval, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))

            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                ids, score = recognizer.predict(gray[y:y+h, x:x+w])
                if (score>46):
                    cv2.putText(frame, 'stranger', (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0),1 )
                else:
                    cv2.putText(frame, str(names[ids-1]), (x+10, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0),1 )

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit()