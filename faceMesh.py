
import cv2
import mediapipe as mp
import time

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

mpDraw = mp.solutions.drawing_utils

drawSpec = mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2) # face er landmark color, radius controll

cap = cv2.VideoCapture('./videos/যেভাবে পুশআপস শিখবো ॥ আবু সুফিয়ান তাজ ॥ আয়মান সাদিক.mp4')
preTime = 0
curTime = 0

while True : 
    _, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # print(results.detections)

    if results.multi_face_landmarks : 
        # for id, face in enumerate(results.multi_face_landmarks) : 

        for faceMl in results.multi_face_landmarks : 
            mpDraw.draw_landmarks(img, faceMl, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

            for id, lm in enumerate(faceMl.landmark) : 
                h, w, c = img.shape

                x, y = int(lm.x * w), int(lm.y * h)
                print(x, y)
   
    
    curTime = time.time()
    fps = 1 / (curTime-preTime)
    preTime = curTime

    cv2.putText(img, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)


    cv2.imshow("image", img)
    cv2.waitKey(1)