FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]

RUN apt update && apt -y upgrade
RUN apt-get upgrade && apt-get -y upgrade
RUN apt install -y sudo git curl make cmake gcc g++

COPY ./archive/python.sh /python.sh
RUN . /python.sh 3.9.16
ENV PATH /usr/local/python/bin:$PATH

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# GTK
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev
# GStreamer
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-doc libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio 
# v4l2 ffmpeg
RUN apt install -y v4l-utils ffmpeg
# Image format
RUN sudo apt-get install -y libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev

COPY ./archive/opencvInstall.sh /opencvInstall.sh
RUN . /opencvInstall.sh

RUN apt-get update \
    && apt-get install -y locales locales-all \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# Camera 
# RUN apt install -y v4l-utils ffmpeg
# Audio
RUN apt install -y portaudio19-dev pulseaudio
# これ入れないとopencvが正常に作動しない
# RUN apt install -y libgl1-mesa-dev

# 画像処理機能
# RUN python -m pip install opencv-python opencv-contrib-python ffmpeg-python
RUN python3 -m pip install pyaudio sounddevice
RUN python3 -m pip install torch torchvision torchaudio
RUN python3 -m pip install tensorflow mtcnn
RUN python3 -m pip install git+https://github.com/openai/whisper.git
RUN python3 -m pip install tqdm
RUN echo 'export QT_X11_NO_MITSHM=1' >> ~/.bashrc

CMD [ "/bin/bash" ]