"""test_3_mac_receive_jpg.py -- receive & display jpg stream.

A simple test program that uses imagezmq to receive an image jpg stream from a
Raspberry Pi and display it as a video steam.

1. Run this program in its own terminal window on the mac:
python test_3_mac_receive_jpg.py

This "receive and display images" program must be running before starting the
RPi sending program.

2. Run the jpg sending program on the RPi:
python test_3_rpi_send_jpg.py

A cv2.imshow() window will appear on the Mac showing the tramsmitted images as
a video stream. You can repeat Step 2 and start the test_3_rpi_send_jpg.py on
multiple RPis and each one will cause a new cv2.imshow() window to open.

To end the programs, press Ctrl-C in the terminal window of the RPi  first.
Then press Ctrl-C in the terminal window of the receiving proram. You may
have to press Ctrl-C in the display window as well.
"""
# import imagezmq from parent directory
import sys
sys.path.insert(0, '../imagezmq')  # imagezmq.py is in ../imagezmq

import numpy as np
import cv2
import imagezmq
import threading

clients = []
index = 0
frames = []
image_hub = imagezmq.ImageHub()


class faceDetection:
    def __init__():
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
    
    def process(image):
        ret = True
        while ret:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(1,1),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            print("Detected {0} faces!".format(len(faces)))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
         
            return image


face = faceDetection.__init__()


def search(value):
    for i, arg in enumerate(clients):
        if( clients[i][0] == value ): 
            return i
    return -1

def process(i):
    while True:
        image = cv2.imdecode(np.fromstring(frames[i], dtype='uint8'), -1)
        #image = face.process(image)
       # cv2.imshow(clients[i][0], image)  
        cv2.waitKey(1)
      #  image_hub.send_reply(b'OK')



        



while True:  # show streamed images until Ctrl-C
    rpi_name, jpg_buffer = image_hub.recv_jpg()
    cond = search(rpi_name)
    if( cond == -1 ):
        clients.append([rpi_name, index])
        print(clients[index])
        print(cond)
        frames.append(jpg_buffer)
       
        image = cv2.imdecode(np.fromstring(jpg_buffer, dtype='uint8'), -1)
        x = threading.Thread(target=process, args=(index,))  
        x.start()
        cv2.imshow("alecs", image)         
        index = index + 1
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')
    else:
        frames[cond] = jpg_buffer
        image = cv2.imdecode(np.fromstring(frames[cond], dtype='uint8'), -1)
        cv2.
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')



        
        
       
    

    # see opencv docs for info on -1 parameter
    #image = face(image)
    # 1 window for each RPi
  




        