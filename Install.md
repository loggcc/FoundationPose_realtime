Setting the environment for foundation pose
```
# Pull the following docker image and launch it:
docker pull shingarey/foundationpose_custom_cuda121:latest

# additional requirements
pip install Hydra fastapi uvicorn

# for real time tracking:
pip install pyrealsense2
pip install ultralytics==8.0.120

```

Build FoundationPose
```
# Enter the container, git clone this repo:
git clone https://github.com/teal024/FoundationPose-plus-plus

# Export project root as the dir of the repo
export PROJECT_ROOT=/root/FoundationPose-plus-plus

# Download FoundationPose weights to $PROJECT_ROOT/FoundationPose/weights
from Google Drive: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i

# Start building process
cd $PROJECT_ROOT/FoundationPose
bash build_all.sh
```

Download Other Weights
```
1. Download Qwen2-VL weights to $PROJECT_ROOT/Qwen2-VL/weights
from huggingface: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct 
2. Download Sam-HQ weights to $PROJECT_ROOT/sam-hq/pretrained_checkpoints
from Google Drive: https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view 
```

Utilities
```
# For SAM-HQ
pip install segment-anything-hq
cd $PROJECT_ROOT/sam-hq
pip install -e .

# For Qwen-VL-2-Instruct
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils

# For Cutie
cd $PROJECT_ROOT/Cutie
pip install -e .
python cutie/utils/download_models.py

```
