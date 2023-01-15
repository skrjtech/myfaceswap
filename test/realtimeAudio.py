import pyaudio
import time

class AudioFilter():
    def __init__(self):
        # オーディオに関する設定
        self.p = pyaudio.PyAudio()
        self.channels = 2 # マイクがモノラルの場合は1にしないといけない
        self.rate = 48000 # DVDレベルなので重かったら16000にする
        self.format = pyaudio.paInt16
        self.stream = self.p.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        output=True,
                        input=True,
                        stream_callback=self.callback)

    # コールバック関数（再生が必要なときに呼び出される）
    def callback(self, in_data, frame_count, time_info, status):
        out_data = in_data
        return (out_data, pyaudio.paContinue)

    def close(self):
        self.p.terminate()

if __name__ == "__main__":
    # AudioFilterのインスタンスを作る場所
    af = AudioFilter()

    # ストリーミングを始める場所
    af.stream.start_stream()

    # ノンブロッキングなので好きなことをしていていい場所
    while af.stream.is_active():
        time.sleep(0.1)

    # ストリーミングを止める場所
    af.stream.stop_stream()
    af.stream.close()
    af.close()