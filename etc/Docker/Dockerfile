FROM tensorflow/tensorflow:2.3.0-gpu as base

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libfontconfig1 \
        git \
        unzip \
        wget \
        vim
    
RUN pip --no-cache-dir install --upgrade \
    pip \
    setuptools \
    opencv-python \
    is-msgs \
    is-wire

RUN pip --no-cache-dir install --upgrade protobuf
# Something funky was going on. This fixes it.
RUN apt install -y --no-install-recommends libgl1-mesa-glx
# Download and convert yolov3.weights
WORKDIR /home
RUN git clone https://github.com/pedrodsc/is-tracker.git -b develop
RUN wget https://pjreddie.com/media/files/yolov3.weights -O /home/is-tracker/etc/data/yolov3.weights
RUN python /home/is-tracker/src/is_tracker/convert.py --weights /home/is-tracker/etc/data/yolov3.weights --output /home/is-tracker/etc/checkpoints/yolov3.tf
RUN python /home/is-tracker/src/is_tracker/export_tfserving.py
#RUN rm /home/is-tracker/etc/data/yolov3.weights
