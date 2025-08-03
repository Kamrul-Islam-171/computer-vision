
import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('./videos/madara.mp4')
preTime = 0
curTime = 0

while True : 
    _, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    print(results.detections)

    if results.detections : 
        for id, detection in enumerate(results.detections) : 
            # print(id, detection)
            # print(id, detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxOc = detection.location_data.relative_bounding_box

            h, w, c = img.shape
            bbox = int(bboxOc.xmin * w), int(bboxOc.ymin * h),  int(bboxOc.width * w), int(bboxOc.height * h) # 4 ponit

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

            # mpDraw.draw_detection(img, detection)    
    
    curTime = time.time()
    fps = 1 / (curTime-preTime)
    preTime = curTime

    cv2.putText(img, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)


    cv2.imshow("image", img)
    cv2.waitKey(1)