
import cv2
import mediapipe as mp
import time

class faceMeshDetector: 
    def __init__(self, mxFace=2):
        self.mxFace = mxFace
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.mxFace)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2) 

    def findFace(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        # print(results.detections)

        faces = []

        if self.results.multi_face_landmarks : 
            # for id, face in enumerate(results.multi_face_landmarks) : 

            for faceMl in self.results.multi_face_landmarks : 
                if draw : 
                    self.mpDraw.draw_landmarks(img, faceMl, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

                face = []    
                for id, lm in enumerate(faceMl.landmark) : 
                    h, w, c = img.shape

                    x, y = int(lm.x * w), int(lm.y * h)
                    # cv2.putText(img, f'{id}', (x,y), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,0,255), 1)
                    # print(x, y)
                    face.append([x, y])
                faces.append(face)    

        return img, faces            
   
    
   


def main() : 


    cap = cv2.VideoCapture('./videos/যেভাবে পুশআপস শিখবো ॥ আবু সুফিয়ান তাজ ॥ আয়মান সাদিক.mp4')
    preTime = 0
    curTime = 0

    detector = faceMeshDetector()

    while True : 
        _, img = cap.read()

        img, faces = detector.findFace(img)
        # print(len(faces))
        # if len(faces) != 0 :
        #     print(faces[0])

        curTime = time.time()
        fps = 1 / (curTime-preTime)
        preTime = curTime

        cv2.putText(img, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)


        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == '__main__': 
    main()    