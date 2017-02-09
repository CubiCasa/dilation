# Multi-Scale Context Aggregation by Dilated Convolutions

## Introduction

Properties of dilated convolution are discussed in our [ICLR 2016 conference paper](http://arxiv.org/abs/1511.07122). This repository contains the network definitions and the trained models. You can use this code together with vanilla Caffe to segment images using the pre-trained models. If you want to train the models yourself, use our [Caffe fork](https://github.com/fyu/caffe-dilation) and please check out the [document for training](https://github.com/fyu/dilation/blob/master/docs/training.md).

# CUBICASA Instructions

Installation of Caffe can be very painful, so if you want to everything from scratch (or have to for some reason). Follow the instructions to get the prediction and training working.

Otherwise just skip all this hassle, and get the readymade Docker image.

## Installation for the prediction

1) start with the [Caffe Docker image](https://github.com/BVLC/caffe/tree/master/docker)

```bash
nvidia-docker run -ti bvlc/caffe:gpu caffe bash
apt-get update
```

2) Install Anaconda 2.xx for the image (check the up-to-date [installer path](https://www.continuum.io/downloads))

```bash
wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
chmod +x Anaconda2-4.3.0-Linux-x86_64.sh 
./Anaconda2-4.3.0-Linux-x86_64.sh 
source /root/.bashrc
```

3) Install dilation dependencies 'conda install numba numpy opencv' 

```bash
conda install numba numpy opencv
```

You will hit probably this error first:

```python
ImportError: /root/anaconda2/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /opt/caffe/python/caffe/_caffe.so)
```

Which can be fixed with:

```bash
 conda install libgcc
```

This is followed by this protobuf error:

```python
ImportError: No module named google.protobuf.internal
```

Which can be again fixed with:

```bash
 conda install protobuf
```

## Install another Caffe for training

Refer to the [document for training](docs/training.md). Now more Caffe fun to come, as we need to clone the forked Caffe of the dilation authors. Note, that we have double Caffe in the docker now hogging up space, and building with [cmake](http://caffe.berkeleyvision.org/installation.html) did not work so you need to modify the `Makefile.config`: 

1) So that PYTHON_INCLUDE refers to the ANACONDA python located in `/root/anaconda2/`
2) HDF5 libary need to be linked later
`INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial`
`LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial`

Fix the **HDF5 library** before advancing with the directions from there: https://github.com/BVLC/caffe/issues/4333

```bash
/usr/bin/ld: cannot find -lhdf5_hl
/usr/bin/ld: cannot find -lhdf5
```

Fix:

```bash
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;
cd /usr/lib/x86_64-linux-gnu
ln -s libhdf5_serial.so.10.1.0 libhdf5.so
ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so 
cd /home/caffe-dilation
```

And then problems with the OpenCV

```bash
Makefile:627: recipe for target 'build_master_release/tools/upgrade_net_proto_text.bin' failed
make: *** [build_master_release/tools/upgrade_net_proto_text.bin] Error 1
```

And the fix is to [modify one line](https://github.com/BVLC/caffe/issues/4621) of `Makefile`

```
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5 \
        opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
```

Finally the actual build should work:

```bash
git clone https://github.com/fyu/caffe-dilation
cd caffe-dilation
apt-get install nano
cp Makefile.config.example Makefile.config && nano Makefile.config
make all
make test
make runtest
```

## Usage

Assuming that we have the functional Docker image now

### Test that prediction works with the example image

```bash
python predict.py pascal_voc images/dog.jpg --gpu 0
```

### Training for new dataset

1) Download the data (from temporary Dropbox path for example)

```bash
mkdir trainData && cd trainData && wget https://www.dropbox.com/s/nd6hjc61h5jujsw/semanticSegmentationLabels.zip?dl=0
apt-get install p7zip-full
7z x semanticSegmentationLabels.zip\?dl\=0 
```

### Extra

#### Viewing images

Install `Feh` for example: https://feh.finalrewind.org/

```bash
apt-get install libcurl4-openssl-dev libx11-dev libxt-dev libimlib2-dev libxinerama-dev libjpeg-progs
git clone git://git.finalrewind.org/feh || git clone git://github.com/derf/feh.git
cd feh
make
make install
```

And then viewing attempt still fails as X Server is not running for the Docker image

```bash
feh images/example_pascal_voc.jpg 
feh ERROR: Can't open X display. It *is* running, yeah?
```

