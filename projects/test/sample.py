import numpy as np

with open("/Database/AudioData/output.wav", "rb") as wavf:
    wav = wavf.read()
# wav = np.frombuffer(wav, np.int8)
for i in range(4):
    res = ''
    for j in range(16):
        index = (i * 16) + j
        res += f"{np.array(wav[index], dtype=np.int8).tobytes()}"
    print(f"{i: 08d}{res}")