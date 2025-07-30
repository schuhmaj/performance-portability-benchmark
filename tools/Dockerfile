FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

USER root

RUN apt update &&  \
    apt install cmake ninja-build gdb nvidia-opencl-dev git libboost-all-dev -y