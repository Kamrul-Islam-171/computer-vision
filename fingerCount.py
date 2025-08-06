
import cv2
import time 
import numpy as np
import mediapipe as mp
import handTrackingModule as htm
import math
import os





wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam) 
cap.set(4, hCam) 


# image gula k list e nilam
folderPath = './images'
myList = os.listdir(folderPath)
print(myList)

overLayList = []
for imPath in myList : 
    image = cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.7)

tipIds = [4, 8, 12, 16, 20]

while True : 
    _, img = cap.read()

    img = cv2.flip(img, 1)
    img = detector.findHands(img)

    lmLists = detector.fintPositions(img, draw=False)

    

    if(len(lmLists) != 0) : 
        #   amra tip landmark gula nibo 5 ta finger er
        #   lanmark 8 jodi landmark 6 er niche tahke taile oi finger ta down

        # thumb, index = lmLists[4], lmLists[8]

        finger = []

        # for thumb tip: 
        if lmLists[tipIds[0]][1] < lmLists[tipIds[0]-1][1] :
            finger.append(1)
        else : finger.append(0)

        for i in range(1,5):

            # if lmLists[8][2] < lmLists[6][2] : 
            if lmLists[tipIds[i]][2] < lmLists[tipIds[i]-2][2] :
                finger.append(1)
            else : finger.append(0)

        # print(finger)        
            
        totalFingers = finger.count(1)    

        h,w,c = overLayList[2].shape
        # img[0:200, 0:200] = overLayList[1] # img er moddhe theke (200 width and 200 height ) er akta space nia hand img k show korbe
        img[0:h, 0:w] = overLayList[totalFingers-1] # img er moddhe theke (200 width and 200 height ) er akta space nia hand img k show korbe

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0,0), 25)

   
  
        

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'fps: {int(fps)}', (400,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0,0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)