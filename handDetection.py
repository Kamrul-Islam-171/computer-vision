import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

preTime = 0
curTime = 0

while True : 
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks) multiple hand

    if results.multi_hand_landmarks : 
        for hand in results.multi_hand_landmarks : 

            for id, landmark in enumerate(hand.landmark):
                h, w, c = img.shape 
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)

                if id == 8 : 
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    curTime = time.time()
    fps = 1 / (curTime-preTime)
    preTime = curTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("images",img)
    cv2.waitKey(1)
     
