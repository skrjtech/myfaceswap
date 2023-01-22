#!/bin/python3
# -*- coding: utf-8 -*-

import cv2
import pyaudio
import numpy as np

def GetChannels(capIdx) -> int:
    cap = cv2.VideoCapture(capIdx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            _, _, c = frame.shape
            cap.release()
            return c
    cap.release()
    return -1

class CaptureBase(object):
    def __init__(self, capIdx) -> None:
        CHANNELS = GetChannels(capIdx)
        if CHANNELS < 0:
            if type(capIdx) == str:
                raise AssertionError(f"videoCapture {capIdx} のファイルを正しく開くことができませんでした")
            else:
                raise AssertionError(f"カメラデバイスID: {capIdx}が正しく開くことができませんでした")
        self.Capture = cv2.VideoCapture(capIdx)
        self.MAXWIDTH = int(self.Capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.MAXHEIGHT = int(self.Capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.MAXFRAMES = int(self.Capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.MAXCHANNELS = CHANNELS
        self.Zeros = np.zeros((self.MAXWIDTH, self.MAXHEIGHT, self.MAXCHANNELS), dtype=np.uint8)
    
    def isOpened(self):
        return self.Capture.isOpened()
    
    def Read(self):
        if self.Capture.isOpened():
            ret, frame = self.Capture.read()
            if not ret:
                frame = self.Zeros.copy()
            return frame
        return self.Zeros.copy()
    
    def imshow(self, frame, name: str='realtime'):
        cv2.imshow(name, frame)
        if cv2.waitKey(1) == ord('q'):
            return False
        return True
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Capture.release()
        cv2.destroyAllWindows()

class AudioBase(object):
    def __init__(self, idx):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=441000, input=True, input_device_index=idx, frames_per_buffer=1024)
    
    def Read(self):
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

class CameraPlugin(CaptureBase):
    def __init__(self, capIdx) -> None:
        super().__init__(capIdx)
    
    def Plugin(self):
        return self.Read()

class AudioPlugin(AudioBase):
    def __init__(self, idx):
        super().__init__(idx)

    def plugin(self):
        return self.Read()

    