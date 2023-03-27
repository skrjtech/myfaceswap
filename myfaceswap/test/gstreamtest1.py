import cv2 as cv
from copy import deepcopy

device = cv.VideoCapture("/dev/video0")

outputDevice = cv.VideoWriter(
    'appsrc ! videoconvert ! videoscale ! video/x-raw,format=I420 ! v4l2sink device=/dev/video10',
    0,           # 出力形式。今回は0で。
    30,          # FPS
    (320, 240),  # 出力画像サイズ
    True,        # カラー画像フラグ
)

while device.isOpened():
    ret, frame = device.read()
    if ret:
        output = deepcopy(frame)
        # output = cv.cvtColor(output, cv.COLOR_BGR2RGB)
        output = cv.resize(output, (320, 240))
        output = cv.flip(output, 1)
        output = cv.putText(output, text="FPS: ", org=(50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
    outputDevice.write(output)