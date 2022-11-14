# image
FROM debian:latest
# init process
RUN apt update && apt -y upgrade
RUN apt-get update && apt-get -y upgrade
# importante package
RUN apt install -y software-properties-common
RUN apt install -y sudo git fonts-takao ffmpeg v4l-utils
# python 
RUN apt install -y python3-pip 
RUN apt install -y build-essential libssl-dev libffi-dev python3-dev
# これ入れないとopencvが正常に作動しない
RUN apt install -y libgl1-mesa-dev
# pip update
RUN python3 -m pip install --upgrade pip setuptools
# 行列統計計算機能 統計計算データベース機能　可視化機能
RUN python3 -m pip install numpy pandas matplotlib
# 画像処理機能
RUN python3 -m pip install opencv-python opencv-contrib-python
# ニューラルネットワーク構築用フレームワーク cpu or gpu (cuda version 11.7 )
RUN python3 -m pip install torch torchvision torchaudio
# MTCNN
RUN python3 -m pip install tensorflow mtcnn
# 進行状況可視化
RUN python3 -m pip install tqdm
# opencv camera 用
RUN echo 'export QT_X11_NO_MITSHM=1' >> ~/.bashrc