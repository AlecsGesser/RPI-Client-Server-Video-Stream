import numpy as np
import cv2 as cv
import imutils



cap = cv.VideoCapture(1)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
fgbg = cv.bgsegm.createBackgroundSubtractorCNT()
fgbg.setMinPixelStability(5)
fgbg.setMaxPixelStability(15*50)
fgbg.setIsParallel(True)
fgbg.setUseHistory(False)

cv.namedWindow('frame',2)
cv.namedWindow('frame_g',2)

ret, model = cap.read()
model = cv.cvtColor(model, cv.COLOR_BGR2GRAY)

while(1):
    ret, frame = cap.read()
    
    #frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame_g = cv.subtract(frame_g, model)
    #ret,frame_g = cv.threshold(frame_g,20,255,cv.THRESH_BINARY)
    frame_blured = frame.copy()
    cv.blur(frame, (11,11), frame_blured)
    fgmask = fgbg.apply(frame_blured)
    for i in range(0,5):
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )
    
    
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
            if point[0][0]  < min_x:  
                min_x = point[0][0]
            if point[0][1] < min_y:  
                min_y = point[0][1]
        cv.rectangle(fgmask, (max_x, max_y), (min_x, min_y), (255,255,255),  cv.FILLED  )






    fgmask_1 = cv.divide(fgmask, 255)
    fgmask_1 = cv.cvtColor(fgmask_1, cv.COLOR_GRAY2BGR)
    frame_blured = cv.multiply(frame, fgmask_1)
    
    cv.imshow('frame',frame_blured)
    cv.imshow('frame_g',fgmask)

    
    
    cv.waitKey(30)
    
cap.release()
cv.destroyAllWindows()