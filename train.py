import os
import cv2
from tkinter import filedialog
from PIL import Image
import numpy as np

def getImgAndLB(path):
    faceSamples = []
    ids = []
    imgpaths = [os.path.join(path,f) for f in os.listdir(path)]
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for imgPath in imgpaths:
        PIL_img=Image.open(imgPath).convert('L')
        img_numpy=np.array(PIL_img,'uint8')
        fc = face_detector.detectMultiScale(img_numpy)
        id = int(os.path.split(imgPath)[1].split('.')[0])
        for x,y,w,h in fc:
            ids.append(id)
            faceSamples.append(img_numpy[y:y+h,x:x+w])
    #print('id:', id)
    #print('fs:', faceSamples)
    return faceSamples, ids

if __name__=='__main__':
    path='C:/Users/95198/Desktop/face'
    fcs, ids=getImgAndLB(path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(fcs,np.array(ids))
    recognizer.write('trainer/trainer.yml')