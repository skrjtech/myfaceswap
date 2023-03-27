import cv2
import numpy

# pipeline = 'gst-launch-1.0 -e qtiqmmfsrc camera-id=0 ! video/x-h264,format=NV12,width=1920,height=1080,framerate=30/1 ! h264parse ! avdec_h264 ! videoconvert ! waylandsink sync=false'

capture = cv2.VideoCapture("/dev/video10")

while cv2.waitKey(1) != 27:
    ret, frame = capture.read()
    cv2.imshow('preview', frame)