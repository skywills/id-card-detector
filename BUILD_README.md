**build opencv docker**
https://qiita.com/kndt84/items/9524b1ab3c4df6de30b8


**build opencv with cuda**
```
sudo apt update
sudo apt upgrade
sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall
sudo apt install libjpeg-dev libpng-dev libtiff-dev
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev 
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev  
sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~
sudo apt-get install libgtk-3-dev
sudo -H pip3 install -U pip numpy
sudo apt install python3-testresources
sudo apt-get install libtbb-dev
sudo apt-get install libatlas-base-dev gfortran
```

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake unzip pkg-config git \
libjpeg-dev libpng-dev libtiff-dev \
libavcodec-dev libavformat-dev libswscale-dev \
libv4l-dev libxvidcore-dev libx264-dev \
libgtk-3-dev \
libatlas-base-dev gfortran 
conda create --name opencv-build python=3.7
conda activate opencv-build
conda install numpy
export python_exec=`which python`
export include_dir=`python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
export library=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`
export default_exec=`which python3.7`
export opencv_install_package="~/anaconda3/envs/opencv-build/lib/python3.7/site-packages"
export opencv_contrib_path="~/Desktop/projects/github/opencv/opencv_contrib"
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
mkdir opencv
cd opencv
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.4.0
cd ..
cd opencv_contrib
git checkout 4.4.0
cd ../opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_ARCH_BIN=6.1 \
-D WITH_CUBLAS=1 \
-D BUILD_TIFF=ON \
-D OPENCV_EXTRA_MODULES_PATH=$opencv_contrib_path/modules/ \
-D HAVE_opencv_python3=ON \
-D PYTHON_EXECUTABLE=$python_exec \
-D PYTHON_DEFAULT_EXECUTABLE=$default_exec \
-D PYTHON_INCLUDE_DIRS=$include_dir \
-D PYTHON_LIBRARY=$library \
-D BUILD_EXAMPLES=ON \
-D CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.7.6.5 \
-D CUDNN_INCLUDE_DIR=/usr/local/cuda/include  \
.. 
```

make -j8
sudo make install
sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
https://github.com/edeane/cards
https://github.com/serengil/tensorflow-101/blob/master/python/face-alignment.py