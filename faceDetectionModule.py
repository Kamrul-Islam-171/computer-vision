
import cv2
import mediapipe as mp
import time

class faceDetector : 
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, img, draw = True) : 

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        # print(self.results.detections)

        bboxs = []

        if self.results.detections : 
            for id, detection in enumerate(self.results.detections) : 
            
                bboxOc = detection.location_data.relative_bounding_box

                h, w, c = img.shape
                bbox = int(bboxOc.xmin * w), int(bboxOc.ymin * h),  int(bboxOc.width * w), int(bboxOc.height * h) # 4 ponit

                bboxs.append([id, bbox, detection.score])

                if draw : 
                    self.fancyDraw(img, bbox, 50, 4)

                if draw : 
                    # cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

            # mpDraw.draw_detection(img, detection)   

        return img, bboxs      
    
    def fancyDraw(self, img, bbox, l, thickness) : 
        x, y, w, h = bbox
        x1, y1 = x + w, y + h # x,y is the left top point and x1, y1 is the right bottom point

        cv2.rectangle(img, bbox, (255, 0, 255), 2)

        #top left
        cv2.line(img, (x, y), (x + l, y), (255, 0, 0), thickness)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 0), thickness)
        #bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 0), thickness)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 0), thickness)
        #top right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 0), thickness)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 0), thickness)
        #bottom right
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 0), thickness)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 0), thickness)

        return img
    


def main() :
    cap = cv2.VideoCapture('./videos/madara.mp4')
    # cap = cv2.VideoCapture(0)
    preTime = 0
    curTime = 0

    detector = faceDetector()

    while True : 
        _, img = cap.read()

        img, bboxs = detector.findFace(img)

        print(bboxs)

        curTime = time.time()
        fps = 1 / (curTime-preTime)
        preTime = curTime

        cv2.putText(img, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)


        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__" : 
    main()