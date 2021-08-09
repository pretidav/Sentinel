import cv2
import pantilthat
from time import sleep
import math
import random

def wakeup():
    pan = 0
    tilt = 0
    pantilthat.pan(pan)
    pantilthat.tilt(tilt)
    return pan, tilt

def standby():
    pantilthat.pan(45)
    pantilthat.tilt(45)

def search(pan,tilt):
    pan = 0
    tilt = 0
    pan += random.randint(-3,3)
    if (pan < 90) or (pan > -90):
        pantilthat.pan(pan)

    tilt += random.randint(-3,3)
    if (tilt < 90) or (tilt > -90):
        pantilthat.tilt(tilt)

    return pan,tilt

def focus(roi,panAngle=0,tiltAngle=0,frame_w=255,frame_h=255):
    (x,y,w,h) = tuple(map(int,roi))
    center_x = x+w/2
    center_y = y+h/2
    #print('X,Y = {},{}'.format(center_x,center_y))
    #print('frame_w, frame_h = {},{}'.format(frame_w,frame_h))
    #print('pan,tilt = {},{}'.format(panAngle,tiltAngle))

    halo = 50

    if (center_x < (frame_w/2 - halo)):
        panAngle -= 1
        if panAngle < -90:
            panAngle = -90
        pantilthat.pan(panAngle)
    if (center_x > (frame_w/2 + halo)):
        panAngle += 1
        if panAngle > 90:
            panAngle = 90
        pantilthat.pan(panAngle)
    if (center_y < (frame_h/2 - halo)):
        tiltAngle -= 1
        if tiltAngle < -90:
            tiltAngle = -90
        pantilthat.tilt(tiltAngle)
    if (center_y > (frame_h/2 + halo)):
        tiltAngle += 1
        if tiltAngle > 90:
            tiltAngle = 90
        pantilthat.tilt(tiltAngle)
    return panAngle, tiltAngle

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

pan, tilt = wakeup() #wakeup servo

# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#tracker
tracker = cv2.TrackerMedianFlow_create()
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

ret, frame = cap.read()
frame = cv2.flip(frame, 0);
roi = face_cascade.detectMultiScale(frame,scaleFactor=1.2, minNeighbors=5) 
for (x,y,w,h) in roi: 
    tracker.init(frame,(x,y,w,h))
count = 0
while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0);
    count += int(ret)
    
    # Update tracker
    success, roi = tracker.update(frame)
    (x,y,w,h) = tuple(map(int,roi))
    
    if success:
        # Tracking success
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        pan, tilt = focus(roi,pan,tilt,width,height)
    else :
        # Tracking failure
        cv2.putText(frame, "ALERT: Detection Failure", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
        if count%5 == 0:
            tracker = cv2.TrackerMedianFlow_create()
            roi = face_cascade.detectMultiScale(frame,scaleFactor=1.2, minNeighbors=5) 
            for (x,y,w,h) in roi: 
                tracker.init(frame,(x,y,w,h))
        #pan,tilt = search(pan,tilt)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

standby()
cap.release()
cv2.destroyAllWindows()
