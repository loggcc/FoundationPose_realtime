import argparse
import os
import torch
import json
import cv2
import sys
import numpy as np
import multiprocessing as mp
from typing import List
import imageio.v2 as imageio  # Suppress DeprecationWarning
import trimesh
from scipy.spatial.transform import Rotation
from VOT import Cutie, Tracker_2D  
from utils.kalman_filter_6d import KalmanFilter6D



src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)


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

# def adjust_pose_to_image_point(
#         ob_in_cam_ori: torch.tensor, 
#         K: np.ndarray, 
#         x: float = -1., 
#         y: float = -1.,
# ) -> np.ndarray:
#     """
#     Adjusts the 6D pose so that its projection matches the given 2D coordinate (x, y).

#     Parameters:
#     - K: Camera intrinsic matrix (3x3).
#     - ob_in_cam: Original 6D pose as a 4x4 transformation matrix.
#     - x, y: Desired 2D coordinates on the image plane.

#     Returns:
#     - ob_in_cam_new: Adjusted 6D pose as a 4x4 transformation matrix.
#     """
#     # Extract rotation (R) and translation (t) from the original pose
#     device = ob_in_cam_ori.device
#     if ob_in_cam_ori.ndim == 3:
#         ob_in_cam = ob_in_cam_ori[0].detach().cpu().numpy()
#     else:
#         ob_in_cam = ob_in_cam_ori.detach().cpu().numpy()
#     R = ob_in_cam[:3, :3]
#     t = ob_in_cam[:3, 3]

#     tx, ty = get_pose_xy_from_image_point(ob_in_cam, K, x, y)

#     # Update the translation vector
#     t_new = np.array([tx, ty, t[2]])

#     # Construct the new transformation matrix with the updated translation
#     ob_in_cam_new = np.eye(4)
#     ob_in_cam_new[:3, :3] = R
#     ob_in_cam_new[:3, 3] = t_new


#     return torch.from_numpy(ob_in_cam_new).to(device)


# def get_pose_xy_from_image_point(
#         ob_in_cam: np.ndarray, 
#         K: np.ndarray, 
#         x: float = -1., 
#         y: float = -1.,
# ) -> np.ndarray:
#     """
#     Adjusts the 6D pose so that its projection matches the given 2D coordinate (x, y).

#     Parameters:
#     - K: Camera intrinsic matrix (3x3).
#     - ob_in_cam: Original 6D pose as a 4x4 transformation matrix.
#     - x, y: Desired 2D coordinates on the image plane.

#     Returns:
#     - ob_in_cam_new: Adjusted 6D pose as a 4x4 transformation matrix.
#     """

#     if x == -1. or y == -1.:
#         return x, y

#     # Extract rotation (R) and translation (t) from the original pose
#     t = ob_in_cam[:3, 3]

#     # Camera intrinsic parameters
#     fx = K[0, 0]
#     fy = K[1, 1]
#     cx = K[0, 2]
#     cy = K[1, 2]

#     # Keep the depth (tz) the same
#     tz = t[2]

#     # Use depth to match the desired 2D point
#     tx = (x - cx) * tz / fx
#     ty = (y - cy) * tz / fy     

#     return tx, ty


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
    # 提取位移 (xyz)
    xyz = pose_arr[:3]
    
    # 提取欧拉角
    euler_angles = pose_arr[3:]
    
    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # 创建 4x4 变换矩阵
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
        rgb_seq_path: str,
        depth_seq_path: str,
        mesh_path: str,
        init_mask_path: str,
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
    # Read the initial mask
    #################################################
    init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
    if init_mask is None:
        print(f"Failed to read mask file {init_mask_path}.")
        return
    init_mask = init_mask.astype(bool)

    #################################################
    # Read the frame list
    #################################################
    frame_color_list = get_sorted_frame_list(rgb_seq_path)
    frame_depth_list = get_sorted_frame_list(depth_seq_path)
    if not frame_color_list or not frame_depth_list:
        print(f"No RGB frames found.")
        return

    #################################################
    # Load the initial frame
    #################################################
    init_frame_filename = frame_color_list[0]
    init_frame_path = os.path.join(rgb_seq_path, init_frame_filename)
    init_frame = cv2.imread(init_frame_path)
    if init_frame is None:
        print(f"Failed to read initial frame.")
        return

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
    import logging

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

    total_frames = len(frame_color_list)
    pose_seq = [None] * total_frames  # Initialize as None
    kf_mean, kf_covariance = None, None

    # Forward processing from initial frame
    for i in range(0, total_frames):
        #################################################
        # Read the frame
        #################################################
        frame_color_filename = frame_color_list[i]
        frame_depth_filename = frame_depth_list[i]
        color = imageio.imread(os.path.join(rgb_seq_path, frame_color_filename))[..., :3]
        color = cv2.resize(color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

        depth = cv2.imread(os.path.join(depth_seq_path, frame_depth_filename), -1) / 1e3
        depth = cv2.resize(depth, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= np.inf)] = 0

        if color is None or depth is None:
            print(f"Failed to read color frame {frame_color_filename} or depth map {frame_depth_filename}")
            continue

        #################################################
        # 6D pose tracking
        #################################################

        if i == 0:
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

            
            # pose is a 4*4 matirx
            

            mask_visualization_color_filename = None
            bbox_visualization_color_filename = None
            if mask_visualization_path is not None:
                os.makedirs(mask_visualization_path, exist_ok=True)
                mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
            if bbox_visualization_path is not None:
                os.makedirs(bbox_visualization_path, exist_ok=True)
                bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)
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
                mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
            if bbox_visualization_path is not None:
                os.makedirs(bbox_visualization_path, exist_ok=True)
                bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)
            if activate_2d_tracker:
                bbox_2d = tracker_2D.track(
                    color,
                    mask_visualization_path=mask_visualization_color_filename,
                    bbox_visualization_path=bbox_visualization_color_filename
                )
            # TODO: get occluded mask
            # adjusted_last_pose = adjust_pose_to_image_point(ob_in_cam=pose, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
            if activate_2d_tracker:
                print(f"Mask visualization path passed to tracker: {mask_visualization_color_filename}")
                print(f"BBox visualization path passed to tracker: {bbox_visualization_color_filename}")
                if not activate_kalman_filter:
                    est.pose_last = adjust_pose_to_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                else:
                    # using kf to estimate the 6d estimation of the last pose
                    kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(est.pose_last))
                    measurement_xy = np.array(get_pose_xy_from_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2))
                    kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                    est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(est.pose_last.device)

	        

            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
            ########### get pose position#############
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
                
            
            
        pose_seq[i] = pose.reshape(4, 4)

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
            cv2.imshow("Live Pose Estimation", vis_color[..., ::-1])  # Convert RGB to BGR for OpenCV
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # vis_depth = draw_posed_3d_box(cam_K, img=depth_colored, ob_in_cam=center_pose, bbox=bbox)
            # vis_depth = draw_xyz_axis(
            #     vis_depth,
            #     ob_in_cam=center_pose,
            #     scale=0.1,
            #     K=cam_K,
            #     thickness=3,
            #     transparency=0,
            #     is_input_rgb=False,
            # )
            
            if not os.path.exists(pose_visualization_path):
                os.makedirs(pose_visualization_path, exist_ok=True)
            
            pose_visualization_color_filename = os.path.join(pose_visualization_path, frame_color_filename)
            imageio.imwrite(
                pose_visualization_color_filename, vis_color
            )
            # pose_visualization_depth_filename = os.path.join(pose_visualization_path, frame_depth_filename)
            # imageio.imwrite(
            #     pose_visualization_depth_filename, vis_depth
            # )
    


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
    
   

    parser.add_argument("--rgb_seq_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/rgb")
    parser.add_argument("--depth_seq_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/depth")
    parser.add_argument("--mesh_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/mesh/$TESTCASE.stl")
    parser.add_argument("--init_mask_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/0_mask.png")
    
    parser.add_argument("--pose_output_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/pose.npy")
    parser.add_argument("--mask_visualization_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/masks_visualization")
    parser.add_argument("--bbox_visualization_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/bbox_visualization")
    parser.add_argument("--pose_visualization_path", type=str, default="/workspace/foundationpose/FoundationPose-plus-plus/$TESTCASE/pose_visualization")
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

    pose_track(
        args.rgb_seq_path,
        args.depth_seq_path,
        args.mesh_path,
        args.init_mask_path,
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
