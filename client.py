import sys
sys.path.insert(0, 'imagezmq')  
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq
import numpy as np


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
fgbg.setMinPixelStability(5)
fgbg.setMaxPixelStability(15)
fgbg.setIsParallel(True)
fgbg.setUseHistory(False)

def process(frame):
    frame_blured = frame.copy()
    cv2.blur(frame, (11,11), frame_blured)
    fgmask = fgbg.apply(frame_blured)
    for i in range(0,5):
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)  
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for contour in contours:
        max_x = -1
        max_y = -1
        min_x = 5000
        min_y = 5000
        for point in contour:
            if point[0][0] > max_x:  
                max_x = point[0][0]
            if point[0][1] > max_y:  
                max_y = point[0][1]
            if point[0][0] < min_x:  
                min_x = point[0][0]
            if point[0][1] < min_y:  
                min_y = point[0][1]
        cv2.rectangle(fgmask, (max_x, max_y), (min_x, min_y), (255,255,255),  cv2.FILLED  )
       
    fgmask_1 = cv2.divide(fgmask, 255)
    fgmask_1 = cv2.cvtColor(fgmask_1, cv2.COLOR_GRAY2BGR)
    frame_blured = cv2.multiply(frame, fgmask_1)
        
    return frame_blured

sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.101:5555')

rpi_name = sys.argv[2]
if sys.argv[1].isdigit(): 
    picam = cv2.VideoCapture(int(sys.argv[1]))
else:
    picam = cv2.VideoCapture(sys.argv[1])

time.sleep(2.0)  
jpeg_quality = 95 # 95 opencv default  
while True:  
    ret, image = picam.read()
    image = process(image)
    ret_code, jpg_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_jpg(rpi_name, jpg_buffer)
