#!/bin/bash

# Stop existing container if running
docker rm -f foundationpose

# Set working directory
DIR=$(pwd)/..

# Allow X11 access
xhost +

# Run the Docker container with RealSense and GPU access (no --rm so it persists)
docker run -it \
  --name foundationpose \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/bus/usb:/dev/bus/usb \
  --device /dev/bus/usb \
  --privileged \
  -v $DIR:/workspace/foundationpose \
  -w /workspace/foundationpose \
  foundationpose:latest

