# # import whisper
 
# # model = whisper.load_model("medium")
# # result = model.transcribe("/Database/AudioData/output.wav")
# # print(result["text"])

# import whisper
# import soundcard as sc
# import threading
# import queue
# import numpy as np
# import argparse

# SAMPLE_RATE = 16000
# INTERVAL = 3
# BUFFER_SIZE = 4096

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', default='base')
# args = parser.parse_args()

# print('Loading model...')
# model = whisper.load_model(args.model)
# print('Done')

# q = queue.Queue()
# b = np.ones(100) / 100

# options = whisper.DecodingOptions()

# def recognize():
#     while True:
#         audio = q.get()
#         if (audio ** 2).max() > 0.001:
#             audio = whisper.pad_or_trim(audio)

#             # make log-Mel spectrogram and move to the same device as the model
#             mel = whisper.log_mel_spectrogram(audio).to(model.device)

#             # detect the spoken language
#             _, probs = model.detect_language(mel)

#             # decode the audio
#             result = whisper.decode(model, mel, options)

#             # print the recognized text
#             print(f'{max(probs, key=probs.get)}: {result.text}')


# th_recognize = threading.Thread(target=recognize, daemon=True)
# th_recognize.start()

# # start recording
# with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
#     audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
#     n = 0
#     while True:
#         while n < SAMPLE_RATE * INTERVAL:
#             data = mic.record(BUFFER_SIZE)
#             audio[n:n+len(data)] = data.reshape(-1)
#             n += len(data)

#         # find silent periods
#         m = n * 4 // 5
#         vol = np.convolve(audio[m:n] ** 2, b, 'same')
#         m += vol.argmin()
#         q.put(audio[:m])

#         audio_prev = audio
#         audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
#         audio[:n-m] = audio_prev[m:n]
#         n = n-m

import os
import tempfile
import threading
import numpy
import pyaudio
import wave
import whisper
import argparse
import queue

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
INTERVAL = 10
BUFFER_SIZE = 4096


print('Loading model...')
model = whisper.load_model('medium')

pa = pyaudio.PyAudio()

stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=BUFFER_SIZE
                 )

print("Start recording")

q = queue.Queue()


def clean():
    stream.stop_stream()
    stream.close()
    pa.terminate()

    while not (q.empty()):
        file_path = q.get()
        os.remove(file_path)


def transcribe():
    file_path = q.get()

    audio = whisper.load_audio(file_path)
    os.remove(file_path)

    audio = whisper.pad_or_trim(audio)

    mean = numpy.mean(numpy.abs(audio))

    print("mean: ", mean)

    if (mean < 0.002):
        print("[Silent]")
        return

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    # decode the audio
    options = whisper.DecodingOptions(fp16=False, language='english')
    result = whisper.decode(model, mel, options)
    print(f'{max(probs, key=probs.get)}: {result.text}')


def gen_text():
    while True:
        transcribe()


def save_audio(data: bytes):
    _, output_path = tempfile.mkstemp(suffix=".wav")
    wf = wave.open(output_path, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    return output_path


threading.Thread(target=gen_text, daemon=True).start()

try:
    buffer = []
    while True:
        n = 0
        while n < RATE * INTERVAL:
            data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            buffer.append(data)
            n += len(data)

        output_path = save_audio(b"".join(buffer))

        q.put(output_path)

        buffer = []


except:
    print("Stop recording")
    clean()
