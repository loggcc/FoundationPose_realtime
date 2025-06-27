#!/bin/bash

# Stop existing container if running
docker rm -f foundationpose
export CUDA_VISIBLE_DEVICES=0

# Set working directory
DIR=$(pwd)/..

# Allow X11 access
xhost +

# Run Docker with full GPU access and NVIDIA runtime
docker run -it \
  --name foundationpose \
  --runtime=nvidia \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env DISPLAY=$DISPLAY \
  --env CUDA_VISIBLE_DEVICES=0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/bus/usb:/dev/bus/usb \
  --device /dev/bus/usb \
  --privileged \
  -v $DIR:/workspace/foundationpose \
  -w /workspace/foundationpose \
  foundationpose:latest

