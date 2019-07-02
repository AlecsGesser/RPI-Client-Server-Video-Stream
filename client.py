import sys
sys.path.insert(0, 'imagezmq')  
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq
import numpy as np

if cv2.__version__ != '4.1.0': 
    print('Opencv version not compatible, please uupdate to the version 4.1.0')
    exit() 



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))  # creating element to the morphologic tranformation at the bgs mask
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()              # creating bgs class
fgbg.setMinPixelStability(5)                                   # setting up parameters
fgbg.setMaxPixelStability(15)
fgbg.setIsParallel(True)
fgbg.setUseHistory(False)

def process(frame):
    frame_blured = frame.copy()     # creating an Mat for the blur result
    cv2.blur(frame, (11,11), frame_blured)
    fgmask = fgbg.apply(frame_blured)	
    for i in range(0,2):
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)  
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,255,255),cv2.FILLED)     
       
    fgmask_1 = cv2.divide(fgmask, 255) # converto an image with value 0-255 for only with 0-1
    frame_blured = cv2.multiply(frame, fgmask_1) # haivng a mask where the area with movements are represented by ones
        
    return frame_blured

if len(sys.argv) == 4 :
    sender = imagezmq.ImageSender(connect_to='tcp://'+sys.argv[3]+':5555')
    rpi_name = sys.argv[2]
    if sys.argv[1].isdigit(): 
        picam = cv2.VideoCapture(int(sys.argv[1]))
        picam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        picam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera selected")
    else:
        picam = cv2.VideoCapture(sys.argv[1])
        picam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        picam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Video file selected")

else: 
    print("Input format:")
    print("              <capture option:(0..1)> - for system cameras and <file_path> for video test>")
    print("              <Client Name>")
    print("              <server IP")
    exit()


cv2.waitKey(1000)   # VideCapture loading
jpeg_quality = 60 # 95 opencv default  
ret, image = picam.read()
while ret:  
    ret, image = picam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = process(image)
    ret_code, jpg_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_jpg(rpi_name, jpg_buffer)
    cv2.waitKey(1)
