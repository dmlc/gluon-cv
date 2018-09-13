# GluonCV C++ Inference Demo

This is a demo application which illustrates how to use existing GluonCV models in c++ environments given exported JSON and PARAMS files. Please checkout [export](https://github.com/dmlc/gluon-cv/tree/master/scripts/deployment/export) for instructions of how to export pre-trained models.

## Demo usage

```bash
# with yolo3_darknet53_voc-0000.params and yolo3_darknet53-symbol.json on disk
./gluoncv-detect yolo3_darknet53_voc demo.jpg
```

![demo](https://user-images.githubusercontent.com/3307514/45458507-d76ff600-b6a8-11e8-92e1-0b1966e4344f.jpg)

Usage:
```
SYNOPSIS
        ./gluoncv-detect <model file> <image file> [-o <outfile>] [--class-file <classfile>] [-e
                         <epoch>] [--gpu <gpu>] [-q] [--no-disp] [-t <thresh>]

OPTIONS
        -o, --output <outfile>
                    output image, by default no output

        --class-file <classfile>
                    plain text file for class names, one name per line

        -e, --epoch <epoch>
                    Epoch number to load parameters, by default is 0

        --gpu <gpu> Which gpu to use, by default is -1, means cpu only.
        -q, --quite Quite mode, no screen output
        --no-disp   Do not display image
        -t, --thresh <thresh>
                    Visualize threshold, from 0 to 1, default 0.3.
```

## Download prebuilt binaries

| Platform                   | Download link |
|----------------------------|---------------|
| Linux(cpu, openblas)       |               |
| Linux(gpu, cuda9.0)        |               |
| Linux(gpu, cuda9.2)        |               |
| Windows x64(cpu, openblas) |               |
| Windows x64(gpu, cuda9.2)  |               |
| Mac OS(apple-blas)         |       x       |

### Tip: how to acquire up to date pre-built libmxnet shared library

1. You can download prebuilt libmxnet binaries from PyPI wheels.
For example, you can download mxnet 1.3.0 wheels from [PyPI](https://pypi.org/project/mxnet/#files), extract libmxnet.{so|dll} by opening the wheel as zip file. For Linux and Mac, the suffix is **so**, for windows, the suffix is **dll**.

2. You can then replace the libmxnet.* in previously downloaded binary package with the extracted libmxnet from PyPI wheel.

By doing this, You can switch between mxnet cpu/gpu or blas versions without build from source.


## Build from source

### Outline

* For all platforms, the first step is to build MXNet from source, with `USE_CPP_PACKAGE = 1`. Details are available on [MXNet website](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU).
* Build cpp inference demo with mxnet cpp-package support.

**We will go through with cpu versions, gpu versions of mxnet are similar but requires `USE_CUDA=1` and `USE_CUDNN=1 (optional)`. See [MXNet website](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU) if interested.**

### Linux
We use Ubuntu as example in Linux section.

1. Install build tools and git
```bash
sudo apt-get update
sudo apt-get install -y build-essential git
# install openblas
sudo apt-get install -y libopenblas-dev
# install opencv
sudo apt-get install -y libopencv-dev
# install cmake
sudo apt-get install -y cmake
```
2. Download MXNet source and build shared library
```bash
cd ~
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CPP_PACKAGE=1
```
3. (optional) Add MXNet to `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=~/incubator-mxnet/lib
```
4. Build demo application
```bash
cd ~
git clone https://github.com/dmlc/gluon-cv.git
cd gluon-cv/scripts/deployment/cpp-inference
mkdir build
cd build
cmake .. -DMXNET_ROOT=~/incubator-mxnet
make -j $(nproc)
```

5. (optional) Copy app to install directory
```bash
make install
# gluoncv-detect and libmxnet.so will be available at ~/gluon-cv/scripts/deployment/cpp-inference/install/
```

### Mac OS

1. Install build tools and git
```bash
# we use homebrew for quick setup, please install dependencies accordingly if homebrew is not available
brew update
brew install cmake
brew install pkg-config
brew install graphviz
brew tap homebrew/core
brew install opencv
# Get pip
easy_install pip
# For visualization of network graphs
pip install graphviz
```

2. Build MXNet shared library
```bash
git clone --recursive https://github.com/apache/incubator-mxnet ~/incubator-mxnet
cd ~/incubator-mxnet
cp make/osx.mk ./config.mk
echo "USE_BLAS = apple" >> ./config.mk
echo "ADD_LDFLAGS += -L/usr/local/lib/graphviz/" >> ./config.mk
echo "USE_CPP_PACKAGE = 1" >> ./config.mk
make -j$(sysctl -n hw.ncpu)
```

3. Build Demo application
```bash
git clone https://github.com/dmlc/gluon-cv.git ~/gluon-cv
cd ~/gluon-cv/scripts/deployment/cpp-inference
mkdir build
cd build
cmake .. -DMXNET_ROOT=~/incubator-mxnet
make -j$(sysctl -n hw.ncpu)
```

4. (optional) Copy app to install directory
```bash
make install
# gluoncv-detect and libmxnet.so will be available at ~/gluon-cv/scripts/deployment/cpp-inference/install/
```

### Windows
1. Install Visual Studio 2015(2017 is good with cpu version, but may not work well with CUDA 9)

2. Install other dependencies(You might need to open cmd with admin permission)
```bash
# chocolatey is convenient and similar to unix package managers(https://chocolatey.org)
choco install -y cmake git wget 7zip
refreshenv
```

3. Download Openblas and Opencv
```bash
# download openblas and extract to c:\openblas for example
wget http://mxnet-files.s3.amazonaws.com/openblas/openblas-0.2.14-x64-install.zip
7z e openblas-0.2.14-x64-install.zip -oc:\openblas
# download opencv
wget https://github.com/opencv/opencv/releases/download/3.4.3/opencv-3.4.3-vc14_vc15.exe
# extract to c:\opencv for example
opencv-3.4.3-vc14_vc15.exe
```

3. Build MXNet shared library
```bash
git clone https://github.com/apache/incubator-mxnet.git c:\incubator-mxnet
cd c:\incubator-mxnet
mkdir build
cd build
set OpenBLAS_HOME=C:\openblas
set OpenCV_DIR=C:\opencv\build
set PATH=%PATH%;C:\openblas;C:\opencv\build\x64\vc14\bin
cmake .. -G "Visual Studio 14 2015 Win64" -DUSE_CUDA=0 -DUSE_CUDNN=0 -DUSE_OPENCV=1 -DUSE_OPENMP=1 -DUSE_PROFILER=1 -DUSE_BLAS=open -DUSE_LAPACK=1 -DUSE_DIST_KVSTORE=0 -DUSE_MKL_IF_AVAILABLE=0 -DUSE_CPP_PACKAGE=1 -DBUILD_CPP_EXAMPLES=0 -DDO_NOT_BUILD_EXAMPLES=1 -DCMAKE_INSTALL_PREFIX=c:\incubator-mxnet
cmake --build . --config "Release" --target INSTALL
xcopy Release\libmxnet.dll ..\lib\
```

4. Build demo application
```bash
git clone https://github.com/dmlc/gluon-cv.git c:\gluon-cv
cd c:/gluon-cv/scripts/deployment/cpp-inference
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64" -DMXNET_ROOT=~/incubator-mxnet
cmake --build . --config "Release" --target INSTALL
# app available at c:\gluon-cv\scripts\deployment\cpp-inference\install\
```
