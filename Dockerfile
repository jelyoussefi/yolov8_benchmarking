FROM openvino/ubuntu20_dev:latest
ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt update -y
RUN apt install -y wget libcanberra-gtk3-module
RUN pip3 install ultralytics==8.0.43

WORKDIR /workspace/
COPY ./utils /workspace/utils
COPY main.py /workspace/

