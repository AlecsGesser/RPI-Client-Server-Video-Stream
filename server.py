import sys
sys.path.insert(0, 'imagezmq')  # imagezmq.py is in ../imagezmq

import numpy as np
import cv2
import imagezmq

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
def process(image):
    ret = True
    while ret:
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(1,1),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        #print("Detected {0} faces!".format(len(faces)))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image

image_hub = imagezmq.ImageHub()

while True:
    rpi_name, jpg_buffer = image_hub.recv_jpg()
    image = cv2.imdecode(np.fromstring(jpg_buffer, dtype='uint8'), -1)
    image = process(image.copy())
    cv2.namedWindow(rpi_name,2)
    cv2.imshow(rpi_name, image)
    cv2.waitKey(40)
    image_hub.send_reply(b'OK')
