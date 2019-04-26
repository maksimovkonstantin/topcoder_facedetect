FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install software-properties-common libsm6 libxext6 libxrender1 nano -y

RUN add-apt-repository ppa:jonathonf/python-3.6
RUN add-apt-repository ppa:jonathonf/python-2.7
RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip
RUN apt-get install -y python2.7 python2.7-dev python-pip

RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

RUN python2.7 -m pip install pip --upgrade
RUN python2.7 -m pip install wheel

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 0

# install mmdetection
RUN pip3 install Cython
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision

COPY mmdetection /mmdetection
WORKDIR /mmdetection
RUN ./compile.sh
RUN python setup.py install

RUN pip3 install opencv-python tqdm pandas jupyter tensorflow-gpu tensorboardX
RUN pip3 install scikit-learn mmcv matplotlib pycocotools six terminaltables

RUN pip install albumentations opencv-python tqdm sklearn pandas mxnet-cu100 easydict

RUN apt-get install -y wget
#set working directory
RUN mkdir /project
COPY insightface /project/insightface
COPY configs /project/configs
COPY prepare_data /project/prepare_data
COPY predict /project/predict
COPY train.sh /project/
COPY test.sh /project/
RUN chmod +x /project/train.sh
RUN chmod +x /project/test.sh


WORKDIR /project
