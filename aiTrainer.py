
import cv2
import time 
import numpy as np
import mediapipe as mp
import poseDetectionModule as ptm
import math
import os





# wCam, hCam = 640, 480


# cap = cv2.VideoCapture('./videos/goUp.mp4')
# cap = cv2.VideoCapture('./videos/pushup.mp4')
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('./videos/kick1.mp4')
# cap.set(3, wCam) 
# cap.set(4, hCam) 



pTime = 0

detector = ptm.poseDetection()

count = 0
direction = 0 # 0 or 1 can accept

while True : 
    _, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img)

    lm_lists = detector.findPositions(img, draw=False)
    # print(lm_lists)

    color = (0, 255, 0)
    if len(lm_lists) != 0: 
        #right hand
        angle = detector.findAngle(img, 12, 14, 16)
        #left hand
        # detector.findAngle(img, 11, 13, 15)

        # for me 70 = lower value and 150 er high
        percentage = int(np.interp(angle, (70, 150), (0, 100)))
        bar = int(np.interp(angle, (70, 150), (650, 100))) # bar er max value 650 and min value 100
        # print(percentage)

        if percentage == 0: 
            direction = 1
            color = (0, 0, 255)
        elif percentage == 100:
            if direction == 1 : 
                count += 1
                direction = 0   

        # print(count)  

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 2 )  
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED )  
        cv2.putText(img, f'{int(percentage)}%', (1050, 70), cv2.FONT_HERSHEY_COMPLEX, 2,color, 4)   

    cv2.rectangle(img, (0, 550), (200, 720), (0, 255, 0), cv2.FILLED )  
    cv2.putText(img, f'{int(count)}', (65, 680), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0,0), 15)         


        

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'fps: {int(fps)}', (400,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0,0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)