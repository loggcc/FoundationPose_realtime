#!/bin/bash

# Stop existing container if running
docker rm -f foundationpose
export CUDA_VISIBLE_DEVICES=0

# Set working directory
DIR=$(pwd)/..

xhost +local:root
docker run --gpus all -it --rm \
    --name foundationpose_ros \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --network=host \
    foundationpose_ros:foxy \
    bash

