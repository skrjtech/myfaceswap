FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt update && apt -y upgrade \
    && apt-get update && apt-get -y upgrade
RUN apt autoclean
RUN apt install -y sudo git fonts-takao

# キャッシュの作成を無効
ENV PYTHONDONTWRITEBYTECODE 1
RUN python -m pip install --upgrade pip setuptools
# 行列統計計算機能 統計計算データベース機能　可視化機能
RUN python -m pip install numpy pandas matplotlib
# 画像処理機能
RUN python -m pip install opencv-python opencv-contrib-python
# ニューラルネットワーク構築用フレームワーク cpu or gpu (cuda version 11.6 )
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# opencv camera 用
RUN echo 'export QT_X11_NO_MITSHM=1' >> ~/.bashrc