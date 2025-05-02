import argparse
import os
import time
import torch
import json
import cv2
import sys
import numpy as np
import multiprocessing as mp
from typing import List
import imageio.v2 as imageio  
import trimesh
from scipy.spatial.transform import Rotation
from VOT import Cutie, Tracker_2D  
from utils.kalman_filter_6d import KalmanFilter6D
import pyrealsense2 as rs
import logging
from ultralytics import YOLO

src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)


def get_object_mask_from_frame(frame, target_class_name, yolo_model, confidence_threshold=0.9, visualize=True):
    frame_h, frame_w = frame.shape[:2]
    mask_full = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # Predict
    results = yolo_model(frame)  # Simplified call

    for result in results:
        classes_names = result.names
        if result.masks is not None:
            masks = result.masks.xy
            for mask, box in zip(masks, result.boxes):
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                conf = float(box.conf[0])

                if class_name.lower() == target_class_name.lower() and conf > confidence_threshold:
                    mask_np = np.array(mask, dtype=np.int32)
                    cv2.fillPoly(mask_full, [mask_np], 255)

                    if visualize:
                        overlay_color = (255, 0, 0)
                        cv2.polylines(frame, [mask_np], isClosed=True, color=overlay_color, thickness=2)
                        cv2.putText(frame, f'{class_name} {conf:.2f}',
                                    (mask_np[0][0], mask_np[0][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)

    if visualize:
        cv2.imshow('YOLO Segmentation', frame)
        cv2.waitKey(1)

    return mask_full

def get_sorted_frame_list(dir: str) -> List:
    files = os.listdir(dir)
    if not files:
        return []
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        return []
    if files[0].count('.') == 1:
        files.sort(key=lambda x: int(x.split('.')[0]))
    elif files[0].count('.') == 2:
        files.sort(key=lambda x: int(x.split('.')[0] + x.split('.')[1]))
    return files


def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor, 
        K: torch.Tensor, 
        x: float = -1., 
        y: float = -1.,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

    Parameters:
    - ob_in_cam: 4x4 pose tensor.
    - K: 3x3 intrinsic matrix tensor.
    - x, y: Desired image coordinates.

    Returns:
    - tx, ty: New x/y in camera coordinate system.
    """

    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y
    
    t = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = t[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty



def project_3d_to_2d(point_3d_homogeneous, K, ob_in_cam):
    # Transform point to camera frame
    point_cam = ob_in_cam @ point_3d_homogeneous

    # Perspective division to get normalized image coordinates
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    # Apply camera intrinsics
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return (int(u), int(v))


def get_mat_from_6d_pose_arr(pose_arr):
    # get (xyz) translation
    xyz = pose_arr[:3]
    
    # get euler angles
    euler_angles = pose_arr[3:]
    
    # generate rotation matirx
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # generate 4*4 tansformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix

def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]


def pose_track(
        mesh_path: str,
        cam_K: np.ndarray,
        pose_output_path: str,
        mask_visualization_path: str,
        bbox_visualization_path: str,
        pose_visualization_path: str,
        est_refine_iter: int,
        track_refine_iter: int,
        activate_2d_tracker: bool = False,
        activate_kalman_filter: bool = False,
):
  
    #################################################
    # Load the mesh
    #################################################
    from FoundationPose.estimater import trimesh_add_pure_colored_texture
    
    mesh_file = os.path.join(mesh_path)
    if not os.path.exists(mesh_file):
        print(f"Mesh file not found.")
        return
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Convert units to meters
    mesh.apply_scale(args.apply_scale)
    if args.force_apply_color:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(args.apply_color), resolution=10)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    #################################################
    # Instantiate the 6D pose estimator
    #################################################

    from FoundationPose.estimater import (
        ScorePredictor,
        PoseRefinePredictor,
        dr,
        FoundationPose,
        logging,
        draw_posed_3d_box,
        draw_xyz_axis,
    )
    

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    logging.info("Estimator initialization done")

    #################################################
    # Instantiate the 2D tracker
    #################################################

    if activate_2d_tracker:     # Default using Cutie as a 2D tracker
        tracker_2D = Cutie()
    else:
        tracker_2D = Tracker_2D()


    #################################################
    # 6D pose tracking
    #################################################

    if activate_kalman_filter:
        kf = KalmanFilter6D(args.kf_measurement_noise_scale)

    pose_seq = [] # Initialize as None
    kf_mean, kf_covariance = None, None
    
    
#################################################################

# realsense pipeline

##################################################################


    # Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)




    try:
        timeout_seconds = 60  # set timeout 1 minute
        start_time = time.time()
        i = None  # Initialize i to ensure it exists

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Apply mask detection
            mask_full = get_object_mask_from_frame(
                frame=frame,
                target_class_name= class_name,  
                yolo_model=yolo_model,
                confidence_threshold=0.9,
                visualize=True 
            )

            init_mask = mask_full.astype(bool)

            if np.any(init_mask):
                print("Mask detected, ready to start pose tracking.")
                i = 0  # Initialize frame counter
                break
            
            if time.time() - start_time > timeout_seconds:
                print("Timeout reached! Failed to detect object.")
                pipeline.stop()
                exit(1)  # exit the program safely
        if i is None:
            raise RuntimeError("Failed to initialize frame counter. Mask detection did not succeed.")

            #################################################
            # 6D pose tracking
            #################################################
            
                
        while True:
            frame = f"{i:06d}.png" # name of the frames
            #################################################
            # Capture frame from RealSense
            
        

            # Create an align object to align depth to color
            

            frames = pipeline.wait_for_frames()

            # Align the depth frame to the color frame
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame: # type: ignore
                print("Failed to get frames")
                continue

            color = np.asanyarray(color_frame.get_data())
            
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # mm to meters

            depth[(depth < 0.001) | (depth >= np.inf)] = 0

            # Optionally resize
            color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

            #################################################
            # 6D pose tracking
            #################################################
            
            if i == 0: #if first frame:
            
                mask = init_mask.astype(np.uint8) * 255
                pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

                pose_arr = get_6d_pose_arr_from_mat(pose)
                position = pose_arr[:3]  # x, y, z in meters
                euler = pose_arr[3:]     # roll, pitch, yaw in radians
                pose_matrix = get_mat_from_6d_pose_arr(pose_arr)  # Convert to 4x4 transformation matrix
                print(f"Frame {i}:")
                print("Position in camera frame (m):", position)
                print("Orientation (Euler XYZ, rad):", euler)
                print("4x4 Transformation Matrix:")
                print(pose_matrix)


                if activate_kalman_filter:
                    kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))

                
            
                mask_visualization_color_filename = None
                bbox_visualization_color_filename = None
                
                if mask_visualization_path is not None:
                    os.makedirs(mask_visualization_path, exist_ok=True)
                    mask_visualization_color_filename = os.path.join(mask_visualization_path, frame)
                if bbox_visualization_path is not None:
                    os.makedirs(bbox_visualization_path, exist_ok=True)
                    bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame)
                if activate_2d_tracker:
                    tracker_2D.initialize(
                        color, 
                        init_info={"mask": init_mask}, 
                        mask_visualization_path=mask_visualization_color_filename, 
                        bbox_visualization_path=bbox_visualization_color_filename
                    )
                    
            else:
                mask_visualization_color_filename = None
                bbox_visualization_color_filename = None
                if mask_visualization_path is not None:
                    os.makedirs(mask_visualization_path, exist_ok=True)
                    mask_visualization_color_filename = os.path.join(mask_visualization_path, frame)
                if bbox_visualization_path is not None:
                    os.makedirs(bbox_visualization_path, exist_ok=True)
                    bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame)
                if activate_2d_tracker:
                    bbox_2d = tracker_2D.track(
                        color,
                        mask_visualization_path=mask_visualization_color_filename,
                        bbox_visualization_path=bbox_visualization_color_filename
                    )
                
                # adjusted_last_pose = adjust_pose_to_image_point(ob_in_cam=pose, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                if activate_2d_tracker:
                    if not activate_kalman_filter:
                        est.pose_last = adjust_pose_to_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                    else:
                        # using kf to estimate the 6d estimation of the last pose
                        kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(est.pose_last))
                        measurement_xy = np.array(get_pose_xy_from_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2))
                        kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                        est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(est.pose_last.device)

                pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
                pose_arr = get_6d_pose_arr_from_mat(pose)
                position = pose_arr[:3]  # x, y, z in meters
                euler = pose_arr[3:]     # roll, pitch, yaw in radians
                pose_matrix = get_mat_from_6d_pose_arr(pose_arr)  # Convert to 4x4 transformation matrix
                print(f"Frame {i}:")
                print("Position in camera frame (m):", position)
                print("Orientation (Euler XYZ, rad):", euler)
                print("4x4 Transformation Matrix:")
                print(pose_matrix)
                if activate_2d_tracker and activate_kalman_filter:
                    # use kf to predict from last pose, and update kf status
                    kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)     # kf is alway one step behind
                
                
            pose_seq.append(pose.reshape(4, 4))
            
    ###################pose visualization       
            if pose_visualization_path is not None:
                # depth_normalized = cv2.normalize(np.clip(depth, 0, 900), None, 0, 255, cv2.NORM_MINMAX)
                # depth_8bit = depth_normalized.astype(np.uint8)
                # depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                center_pose = pose @ np.linalg.inv(to_origin)
                vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis_color = draw_xyz_axis(
                    vis_color,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=cam_K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
                # Show visualization live
                cv2.imshow("Live Pose Estimation", cv2.cvtColor(vis_color, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if not os.path.exists(pose_visualization_path):
                    os.makedirs(pose_visualization_path, exist_ok=True)
                
                pose_visualization_color_filename = os.path.join(pose_visualization_path, frame)
                imageio.imwrite(pose_visualization_color_filename, vis_color[..., ::-1])  # Convert BGR to RGB for saving
                
                
            i=i+1
                
            
                

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        pipeline.stop()



    #################################################
    # Save pose sequence
    #################################################

    pose_seq_array = np.array(pose_seq)
    np.save(
        pose_output_path, pose_seq_array
    )

    # Clear GPU memory
    torch.cuda.empty_cache()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
   
    parser.add_argument("--mesh_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/mesh/$TESTCAS.stl")
    parser.add_argument("--pose_output_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/pose.npy")
    parser.add_argument("--mask_visualization_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/masks_visualization")
    parser.add_argument("--bbox_visualization_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/bbox_visualization")
    parser.add_argument("--pose_visualization_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/pose_visualization")
    parser.add_argument("--cam_K", type=json.loads, default="[[387.88845825, 0.0, 323.28192139], [0.0, 387.46902466, 237.11705017], [0.0, 0.0, 1.0]]", help="Camera intrinsic parameters")
    parser.add_argument("--est_refine_iter", type=int, default=10, help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=5, help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05, help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=0.01, help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]", help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color")
    args = parser.parse_args()
    yolo_model_path = '/workspace/foundationpose/FoundationPose-plus-plus/best.pt'
    yolo_model = YOLO(yolo_model_path)  
    class_name = "blue_tube"

    pose_track(args.mesh_path,
        np.array(args.cam_K),
        args.pose_output_path,
        args.mask_visualization_path,
        args.bbox_visualization_path,
        args.pose_visualization_path,
        args.est_refine_iter,
        args.track_refine_iter,
        args.activate_2d_tracker,
        args.activate_kalman_filter,
    )

    torch.cuda.empty_cache()
