# Realtime FoundationPose: Real-Time 6D Object Tracking with YOLOv8 and RealSense

This project extends FoundationPose++ by enabling real-time 6D pose tracking using a YOLOv8 segmentation model and Intel RealSense RGB-D camera. It eliminates the need for pre-recorded mask inputs by performing online object segmentation based on class name, using only the object’s mesh model as prior. 

[FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus.git) is built on top of [FoundationPose](https://github.com/NVlabs/FoundationPose) consists of four main modules: FoundationPose + 2D Tracker + Kalman Filter + Amodal Completion.


FoundationPose is a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups.Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. 

<img src="assets/intro.jpg" width="70%">

<img src="assets/pipeline.jpg" width="70%">



## Method -- Realtime 6D object tracking

This implementation enhances FoundationPose++ with a real-timeprocessing pipeline:



This project extends FoundationPose++ for real-time 6D object pose tracking using only the object’s 3D mesh model as input. Our pipeline operates directly on frames from a RealSense RGB-D camera and enhances FoundationPose++ with an automated segmentation and simplified pose tracking mechanism.
Pose Estimation Pipeline:

- **Mask Segmentation(For online mode only)**:
  - To automate the object mask generation required by FoundationPose++, we integrate a YOLOv8 segmentation model. Given a class name (e.g., "blue_tube"), YOLOv8 detects and segments the object directly from the RealSense color frame. The resulting binary mask is used to initialize the pose tracking module, eliminating manual preprocessing and making the system deployable in fully autonomous setups.
  - A YOLOv8 segmentation model detects and segments the target object in the live RGB stream from a RealSense camera, using only the class name as a reference.
  - This replaces manual mask setup, enabling fully autonomous tracking.


- **Tracking**:
  - **(x, y)**: Estimated using a lightweight 2D tracker (e.g., Cutie, OSTrack) on the segmented region.
  - **z (depth)**: Extracted directly from the RealSense depth map at the (x, y) pixel location.
  - **(roll, pitch, yaw)**: Tracked using a simple Kalman Filter, initialized from FoundationPose++'s rotational predictions and smoothed across frames.

This method leverages a modular engineering approach for 6D pose estimation and achieves accurate, real-time performance without requiring multi-view inputs or offline refinement.

<br>


## Demos

### Realtime Manipulation application

![Realtime Demo](assets/realtime_tracking.gif)

### offline demo for book

![offline Demo](assets/book.gif)





## Environment Setup
Check [install.md](./Install.md) to install all the dependencies.

<br>

## Data prepare


1) [Download demo data](https://drive.google.com/drive/folders/1d5r2kKmLp0LrwcIyJ-ldvjb4zBZkXhIE?usp=drive_link) and extract them under the folder `test_data/` and `test_realtime/`

2) [Optional] Download our customized YOLOv8 segmentation model best.pt



- offline mode
Your testcase data should be formatted like:
```
$PROJECT_ROOT/testdata/$TESTCASE
└── color
    ├── 0.png
    ├── 1.png
    └── ...
└── depth
    ├── 0.png
    ├── 1.png
    └── ...
└── mesh
    ├── mesh.obj/mesh.stl/etc.
    
└── 0_mask.png
```
There should be an RGB image file and a corresponding depth file for each frame, as well as a mesh file of the object, following [FoundationPose](https://github.com/NVlabs/FoundationPose) data format. You can check out [FoundationPose_manual](https://github.com/030422Lee/FoundationPose_manual) if you are not familiar with FoundationPose.




- online mode


```
1. Train customized YOLOv8 segmentaion model for your objects using Roboflow’s guide: With the instruction in roboflow: https://blog.roboflow.com/how-to-train-yolov8-instance-segmentation/


2. After training, save the resulting model (e.g., best.pt) under the path: $PROJECT_ROOT


3. Place the mesh model in the following structure:


$PROJECT_ROOT/test_realtime/$TESTCASE

└── mesh
    ├── mesh.obj/mesh.stl/etc.
```

<br>



## Run demo

###  Run offline demo:
```

export TESTCASE="book"
python src/obj_pose_track.py \
--rgb_seq_path /workspace/foundationpose/test_data/$TESTCASE/color \
--depth_seq_path /workspace/foundationpose/test_data/$TESTCASE/depth \
--mesh_path /workspace/foundationpose/test_data/$TESTCASE/mesh/$TESTCASE.stl \
--init_mask_path /workspace/foundationpose/test_data/$TESTCASE/0_mask.png \
--pose_output_path /workspace/foundationpose/test_data/$TESTCASE/pose.npy \
--mask_visualization_path /workspace/foundationpose/test_data/$TESTCASE/mask_visualization \
--bbox_visualization_path /workspace/foundationpose/test_data/$TESTCASE/bbox_visualization \
--pose_visualization_path /workspace/foundationpose/test_data/$TESTCASE/pose_visualization \
--cam_K "[[387.88845825, 0.0, 323.28192139], [0.0, 387.46902466, 237.11705017], [0.0, 0.0, 1.0]]" \
--activate_2d_tracker \
--apply_scale 0.01 \
--force_apply_color \
--apply_color "[0, 159, 237]" \
--est_refine_iter 10 \


```
<br>

### Run realtime demo

```
export TESTCASE="testtube_realtime"
python src/obj_track_realtime.py \
--mesh_path /workspace/foundationpose/test_realtime/$TESTCASE/mesh/$TESTCASE.stl \
--pose_output_path /workspace/foundationpose/test_realtime/$TESTCASE/pose.npy \
--mask_visualization_path /workspace/foundationpose/test_realtime/$TESTCASE/mask_visualization \
--bbox_visualization_path /workspace/foundationpose/test_realtime/$TESTCASE/bbox_visualization \
--pose_visualization_path /workspace/foundationpose/test_realtime/$TESTCASE/pose_visualization \
--cam_K "[[387.88845825, 0.0, 323.28192139], [0.0, 387.46902466, 237.11705017], [0.0, 0.0, 1.0]]" \
--activate_2d_tracker \
--apply_scale 0.01 \
--force_apply_color \
--apply_color "[0, 159, 237]" \
--est_refine_iter 10 \

```
<br>

