# Build using the following command in the root folder
#   docker build -t cuda-dev .
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

LABEL author="Jonas Schuhmacher"
LABEL email="jonas.schuhmacher@tum.de"

USER root

RUN apt update
RUN apt install cmake ninja-build gdb nvidia-opencl-dev git libboost-all-dev -y