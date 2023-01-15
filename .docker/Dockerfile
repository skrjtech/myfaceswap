ARG FROMIMAGE
FROM skrjtech/python:${FROMIMAGE}

RUN apt-get update \
    && apt-get install -y locales locales-all \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Camera Sound
RUN apt install -y v4l-utils ffmpeg

# これ入れないとopencvが正常に作動しない
RUN apt install -y libgl1-mesa-dev

# 画像処理機能
RUN python -m pip install opencv-python opencv-contrib-python

# Audio
RUN apt install -y portaudio19-dev pulseaudio
RUN python -m pip install pyaudio ffmpeg-python
# ニューラルネットワーク構築用フレームワーク cpu or gpu (cuda version 11.7 )
RUN python -m pip install torch torchvision torchaudio
# MTCNN
RUN python -m pip install tensorflow mtcnn
# NVIDIA WHISPER
RUN python -m pip install git+https://github.com/openai/whisper.git
# 進行状況可視化
RUN python -m pip install tqdm
# opencv camera 用
RUN echo 'export QT_X11_NO_MITSHM=1' >> ~/.bashrc
CMD [ "/bin/bash" ]