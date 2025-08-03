
import cv2
import mediapipe as mp
import time

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

        lm_lists = []

        if self.results.pose_landmarks: 
            for id, lm in enumerate(self.results.pose_landmarks.landmark) : 
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lm_lists.append([id, cx, cy])
                if draw :
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return lm_lists    
    
    


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