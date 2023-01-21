#!/bin/python3
# -*- coding: utf-8 -*-

import cv2
from utils.types import *
import pyaudio

class CameraPlugin(object):
    def __init__(self, camIdx: CaptureT):
        self.Capture = Cap = cv2.VideoCapture(camIdx)
        self.MAXWIDTH = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.MAXHEIGHT = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Capture.release()
        cv2.destroyAllWindows()

    def isOpened(self):
        return self.Capture.isOpened()

    def Plugin(self):
        if self.Capture.isOpened():
            ret, frame = self.Capture.read()
            if not ret:
                frame = np.zeros((self.MAXWIDTH, self.MAXHEIGHT, 3))
        else:
            frame = np.zeros((self.MAXWIDTH, self.MAXHEIGHT, 3))
        return frame

    def imshow(self, frame, name: str='realtime'):
        cv2.imshow(name, frame)
        cv2.waitKey(1)

class AudioPlugin(object):
    def __init__(self, idx):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=441000, input=True, input_device_index=idx, frames_per_buffer=1024)

    def plugin(self):
        datas = []
        for i in range(int((1 / (1 / 441000)) / 1024)):
            frame = self.stream.read(1024)
            datas.append(frame)
        data = b''.join(datas)
        data = np.frombuffer(data, dtype=np.int16)
        return data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()