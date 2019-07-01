import sys
sys.path.insert(0, 'imagezmq')  
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
fgbg.setMinPixelStability(5)
fgbg.setMaxPixelStability(15)
fgbg.setIsParallel(True)
fgbg.setUseHistory(False)

def process(frame):
    frame_blured = frame.copy()
    cv2.blur(frame, (11,11), frame_blured)
    fgmask = fgbg.apply(frame_blured)	
    for i in range(0,2):
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)  
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,255,255),cv2.FILLED)     
       
    fgmask_1 = cv2.divide(fgmask, 255)
    frame_blured = cv2.multiply(frame, fgmask_1)
        
    return frame_blured

sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.239:5555')

rpi_name = sys.argv[2]
if sys.argv[1].isdigit(): 
    picam = cv2.VideoCapture(int(sys.argv[1]))
    picam.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    picam.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)
else:
    picam = cv2.VideoCapture(sys.argv[1])


time.sleep(2.0)  
jpeg_quality = 50 # 95 opencv default  
while True:  
    ret, image = picam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = process(image)
    ret_code, jpg_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_jpg(rpi_name, jpg_buffer)
    cv2.waitKey(40)
