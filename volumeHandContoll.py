
import cv2
import time 
import numpy as np
import mediapipe as mp
import handTrackingModule as htm
import math

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

#pip install pycaw for volume

#########################

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
# volume.SetMasterVolumeLevel(0, None)

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

wCam, hCam = 640, 480
# wCam, hCam = 1280, 720

#########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam) # set the width of camera
cap.set(4, hCam) # set the height of camera

pTime = 0

detector = htm.handDetector(detectionCon=0.7)
volBr = 400
volPercentange = 0

while True : 
    _, img = cap.read()

    img = cv2.flip(img, 1)

    img = detector.findHands(img)

    lmLists = detector.fintPositions(img, draw=False)

    if(len(lmLists) != 0) : 
        # print(lmLists[2]) # if i need info of point 2 or landmark 2
        thumb, index = lmLists[4], lmLists[8]

        midX, midY = (thumb[1] + index[1]) // 2, (thumb[2] + index[2]) // 2 # floor division (returns an integer by discarding the decimal part)

        cv2.circle(img, (thumb[1], thumb[2]), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (index[1], index[2]), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (midX, midY), 15, (255, 0, 255), cv2.FILLED)

        #draw a line between them
        cv2.line(img, (thumb[1], thumb[2]), (index[1], index[2]), (255, 0, 255), 2)
        # cv2.line(img, (thumb[1], thumb[2]), (index[1], index[2]), (255, 0, 255), 2)

        #hypot calculate euclidian distance between two ponits
        length = math.hypot(thumb[1] - index[1], thumb[2] - index[2]) # hypot(x2-x1, y2-y1)
        # print(length)

        # my hand range 50-260
        # volume Range -65 t0 0
        # so we need to convert hand range into volume range

        vol = np.interp(length, [50,260], [minVol, maxVol])
        volBr = np.interp(length, [50,260], [400, 150])
        volPercentange = np.interp(length, [50,260], [0, 100]) # for percentage
        ######################################################
        # interp = do linear interpolation. akta value k ak range theke arek range e nia jay
        # suppose x = 5 hocche 1 - 200 range e.
        # akhon ami chai j x er value ta 500 - 800 range hoile koto hoito
        ######################################################
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50 : 
            cv2.circle(img, (midX, midY), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)        
    cv2.rectangle(img, (50, int(volBr)), (85, 400), (0, 255, 0), cv2.FILLED) 
    cv2.putText(img, f'{int(volPercentange)} %', (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255,0), 2)       

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'fps: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0,0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)