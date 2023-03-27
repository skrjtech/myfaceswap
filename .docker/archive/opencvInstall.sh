#!/bin/bash

pip install numpy

sudo apt install -y wget

mkdir OpenCV
cd OpenCV
wget https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz
tar -zxvf 4.7.0.tar.gz
rm -f 4.7.0.tar.gz
cd opencv-4.7.0
# mkdir build
# cd build
# # cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -DINSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D PYTHON_EXECUTABLE=/usr/bin/python3 -D BUILD_EXAMPLES=ON -D WITH_GTK=ON -D WITH_GSTREAMER=ON -D WITH_FFMPEG=OFF -D WITH_QT=OFF ..
# cmake ..
# make -j 8
# sudo make install 

# git clone https://github.com/opencv/opencv.git
# cd opencv/
# git checkout 4.1.0

mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=/usr/local/python/bin/python3 \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=/usr/local/python \
-D PYTHON3_EXECUTABLE=/usr/local/python/bin/python3 \
-D PYTHON3_INCLUDE_DIR=/usr/local/python/include/python3.9 \
-D PYTHON3_PACKAGES_PATH=/usr/local/python/lib/python3.9/site-packages \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..

sudo make -j$(nproc)
sudo make install
sudo ldconfig