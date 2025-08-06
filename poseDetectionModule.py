
import cv2
import mediapipe as mp
import time
import math

class poseDetection: 
    def __init__(self, mode=False, upBody=False, smooth=True, detCon=0.5, tracCon=0.5, mComplexity=1, enSegment=False):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detCon = detCon
        self.mComplexity = mComplexity
        self.enSegment = enSegment
        self.tracCon = tracCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.mComplexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.enSegment,
            min_detection_confidence=self.detCon,
            min_tracking_confidence=self.tracCon
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)
     
        if self.results.pose_landmarks : 
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS )

        return img

    def findPositions(self, img, draw=True) :

        self.lm_lists = []

        if self.results.pose_landmarks: 
            for id, lm in enumerate(self.results.pose_landmarks.landmark) : 
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                self.lm_lists.append([id, cx, cy])
                if draw :
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return self.lm_lists    

    def findAngle(self, img, p1, p2, p3, draw=True) :
        # p1, p2, p3 are point1 point2 and point3
        x1, y1 = self.lm_lists[p1][1:] # [4, 404, 333] 4 hocce koto no lnadmark, 404 and 333 hocce x and y 
        x2, y2 = self.lm_lists[p2][1:] 
        x3, y3 = self.lm_lists[p3][1:]

        # find angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0 : angle += 360

        if draw : 
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            # cv2.putText(img, str(int(angle)), (x2 - 10, y2 + 60), cv2.FONT_HERSHEY_PLAIN, 3, (100,155,255), 3)

        return angle    
    
    


def main() : 

    cap = cv2.VideoCapture('./videos/jjk.mp4')
    preTime = 0
    curTime = 0

    detector = poseDetection()
    while True : 
        _, img = cap.read()
        img = detector.findPose(img)
        landmarks = detector.findPositions(img, draw=False)


        lp = 14
        if len(landmarks) != 0:
            print(landmarks[lp])

            cv2.circle(img, (landmarks[lp][1], landmarks[lp][2]), 10, (0, 0, 255), cv2.FILLED)    

        curTime = time.time()
        fps = 1 / (curTime-preTime)
        preTime = curTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == '__main__' : 
    main()