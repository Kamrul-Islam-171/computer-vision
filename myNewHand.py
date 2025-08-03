import cv2 
import mediapipe as mp
import time
import handTrackingModule as htm

preTime = 0
curTime = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True : 
    success, img = cap.read()
    img = cv2.flip(img, 1)

 

    img = detector.findHands(img)

    lm_lists = detector.fintPositions(img, draw=False)
    if len(lm_lists) != 0 :
        print(lm_lists[4]) # 4 no landmarks
        

    curTime = time.time()
    fps = 1 / (curTime-preTime)
    preTime = curTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("images",img)
    cv2.waitKey(1)
