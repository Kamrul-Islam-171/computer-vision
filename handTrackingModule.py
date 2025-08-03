import cv2 
import mediapipe as mp
import time


class handDetector : 
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # self.mode = mode
        # self.maxHands = maxHands
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        # self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        # self.mpDraw = mp.solutions.drawing_utils
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(      
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True) :    

         # print(results.multi_hand_landmarks) multiple hand
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks : 
            for hand in self.results.multi_hand_landmarks : 

                if draw : 
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

        return img            

    def fintPositions(self, img, handNo=0, draw=True) : 
        lm_list = []

        if self.results.multi_hand_landmarks : 
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                h, w, c = img.shape 
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)

                lm_list.append([id, cx, cy])

                if draw : 
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        
        return lm_list

     

def main() : 
    preTime = 0
    curTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True : 
        success, img = cap.read()
        img = cv2.flip(img, 1)

 

        img = detector.findHands(img)

        lm_lists = detector.fintPositions(img)
        if len(lm_lists) != 0 :
            print(lm_lists[4]) # 4 no landmarks
        

        curTime = time.time()
        fps = 1 / (curTime-preTime)
        preTime = curTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("images",img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()