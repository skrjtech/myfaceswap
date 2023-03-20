# import speech_recognition as sr
 
# r = sr.Recognizer()
 
# with sr.AudioFile("/Database/AudioData/output.wav") as source:
#     print(source.stream)
#     audio = r.record(source)
 
# text = r.recognize_google(audio, language='ja-JP')
 
# print(text)

# import time
# import speech_recognition as sr

# def callback(recognizer, audio):
#     try:
#         print("result: " + recognizer.recognize_google(audio, language='ja-JP'))
#     except sr.UnknownValueError:
#         print("Google Speech Recognition could not understand audio")
#     except sr.RequestError as e:
        # print("Could not request results from Google Speech Recognition service; {0}".format(e))


# r = sr.Recognizer()
# m = sr.Microphone()
# with m as source:
#     r.adjust_for_ambient_noise(source)
# stop_listening = r.listen_in_background(m, callback)
# # do some unrelated computations for 5 seconds
# for _ in range(50): time.sleep(0.1)
# stop_listening(wait_for_stop=False)

# r = sr.Recognizer()
# while True:
#     try:
#         m = sr.Microphone()
#         with m as source:
#             r.adjust_for_ambient_noise(source)
#         stop_listening = r.listen_in_background(m, callback)
#         for _ in range(50): time.sleep(0.1)
#         stop_listening(wait_for_stop=False)
#     except KeyboardInterrupt:
#         break
# import wave 
# import torch
# import soundfile
# import speech_recognition as sr
# from espnet_model_zoo.downloader import ModelDownloader
# from espnet2.bin.asr_inference import Speech2Text

# d = ModelDownloader()
# speech2text = Speech2Text(
#         **d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave"),
#         device="cpu"  # CPU で認識を行う場合は省略
#         )
# speech2text = Speech2Text.from_pretrained(
#     "kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave"
# )

# speech2text = Speech2Text.from_pretrained(
#     "kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave"
# )

# r = sr.Recognizer()
# with sr.Microphone(sample_rate=16_000) as source:
#     print("なにか話してください")
#     audio_data = r.listen(source)

# frame_bytes = audio_data.get_raw_data()
# speech_array = np.frombuffer(frame_bytes, dtype=np.int16)

# nbests = speech2text(speech_array)
# text, tokens, *_ = nbests[0]
# print(text)

import pyaudio
import numpy as np
import whisper
import wave
 
model = whisper.load_model("medium")
# result = model.transcribe("/Database/AudioData/output.wav")
# print(result["text"])

# CHUNK=1024
CHUNK=1024
RATE=44100
p=pyaudio.PyAudio()
device = 0 #マイクの選択
chunk = 1024
format = pyaudio.paInt16 #フォーマットは16bit
channel = 1 #モノラルで録音
rate = 44100 #サンプリングレート
stream=p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    frames_per_buffer=CHUNK,
    input_device_index=0,
    input=True,
    output=False
) # inputとoutputを同時にTrueにする

def CallBack(data):
    result = model.transcribe("/Database/AudioData/output.wav")
    print(result["text"])
    return data

while stream.is_active():
    print("SPEECH")
    frames = []
    for _ in range(int(RATE // CHUNK) * 3):
        input = stream.read(CHUNK, exception_on_overflow=False)
        # output = stream.write(input)
        frames.append(input)
    wf = wave.open('/Database/AudioData/output.wav', 'wb') # ファイルに保存するよ
    wf.setnchannels(channel)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close() #ファイルを保存したよ
    data = CallBack(frames)

stream.stop_stream()
stream.close()
p.terminate()