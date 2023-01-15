import pyaudio  #pyaudioをインポート
import wave     #データ保存用にwaveを使う
import time

device = 0 #マイクの選択
chunk = 1024
format = pyaudio.paInt16 #フォーマットは16bit
channel = 1 #モノラルで録音
rate = 44100 #サンプリングレート
times = 5 #10秒間録音する
output_path = '/Database/AudioData/output.wav' #保存先の名前

p = pyaudio.PyAudio() #録音するんやで
stream = p.open(format = format,
                channels = channel,
                rate = rate,
                input = True,
                input_device_index = device,
                frames_per_buffer = chunk) #ストリームを開いて録音開始!
print("now recoding...")

frames = [] #録音したデータをしまうList
for i in range(0, int(rate / chunk * times)):
  data = stream.read(chunk,
                exception_on_overflow=False)
  frames.append(data)

print('done.') #録音終わったよ！

stream.stop_stream() #用済みどもの始末
stream.close()
p.terminate()

wf = wave.open(output_path, 'wb') # ファイルに保存するよ
wf.setnchannels(channel)
wf.setsampwidth(p.get_sample_size(format))
wf.setframerate(rate)
wf.writeframes(b''.join(frames))
wf.close() #ファイルを保存したよ