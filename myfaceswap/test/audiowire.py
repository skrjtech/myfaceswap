# -*- coding:utf-8 -*-

import pyaudio
import numpy as np

CHUNK=1024
RATE=44100
p=pyaudio.PyAudio()

stream=p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    frames_per_buffer=CHUNK,
    input_device_index=0,
    input=True,
    output=True
) # inputとoutputを同時にTrueにする

def CallBack(data):
    data = np.frombuffer(data, dtype="int16") / 32768.
    print(f'max: {np.max(data):3.3f} min: {np.min(data):3.3f}\r', end='')
    data = (data * 1.25).astype(np.int16).tobytes()
    return data

while stream.is_active():
    input = stream.read(CHUNK, exception_on_overflow=False)
    data = CallBack(input)
    # output = stream.write(input)
    output = stream.write(data)

stream.stop_stream()
stream.close()
p.terminate()