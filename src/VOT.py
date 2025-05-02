import os
import sys
import argparse
import json
import torch
import numpy as np
from typing import List, Dict
import multiprocessing as mp
import cv2
from torchvision.transforms.functional import to_tensor
from PIL import Image

from utils import visualize_mask, visualize_bbox

proj_path = os.path.join(os.path.dirname(__file__), '..')
cutie_path = os.path.join(proj_path, "Cutie")
if cutie_path not in sys.path:
    sys.path.append(cutie_path)


class Tracker_2D:
    def __init__(self):
        pass

    def initialize(
            self, 
            frame: np.ndarray,
            init_info: Dict[str, np.ndarray], # mask, bbox, etc.  
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    ):
        return [-1, -1, 0, 0]

    def track(
            self, 
            frame: np.ndarray,
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    ):
        return [-1, -1, 0, 0]


class Cutie(Tracker_2D):
    def __init__(
            self,
            cutie_seg_threshold: float = 0.1,
            erosion_size: int = 5,
    ):
        super().__init__()
        self.cutie_seg_threshold = cutie_seg_threshold
        self.erosion_size = erosion_size

        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model
        # Initialize cutie_processor here
        self.cutie = get_default_model()
        self.cutie_processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.cutie_processor.max_internal_size = -1

    def initialize(
            self, 
            init_frame: np.ndarray,
            init_info: Dict[str, np.ndarray], # mask, bbox, etc.  
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    )->List[int]:
        with torch.no_grad():
            init_frame_tensor = to_tensor(init_frame).cuda().float()
            
            init_mask_tensor = torch.from_numpy(init_info['mask']).cuda()
            
            objects = np.unique(init_info['mask'])
            # background "0" does not count as an object
            objects = objects[objects != 0].tolist()
            
            output_prob = self.cutie_processor.step(init_frame_tensor, init_mask_tensor, objects=objects)
            
            # convert output probabilities to an object mask
            mask = self.cutie_processor.output_prob_to_mask(output_prob, segment_threshold=self.cutie_seg_threshold)

            mask_np = mask.cpu().numpy()

        bbox_xywh = self._parse_output(mask_np)

        init_frame = init_frame.copy()

        if mask_visualization_path is not None:
            visualize_mask(init_frame, mask_np*255, mask_visualization_path)
        if bbox_visualization_path is not None:
            visualize_bbox(init_frame, bbox_xywh, bbox_visualization_path)

        # del init_frame_tensor, init_mask_tensor, output_prob
        torch.cuda.empty_cache()

        return bbox_xywh

    def track(
            self, 
            frame: np.ndarray, 
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    ) -> List[int]:
        with torch.no_grad():
            frame_tensor = to_tensor(frame).cuda().float()
            output_prob = self.cutie_processor.step(frame_tensor)

            # convert output probabilities to an object mask
            mask = self.cutie_processor.output_prob_to_mask(output_prob, segment_threshold=self.cutie_seg_threshold)

            mask_np = mask.cpu().numpy()
            
        bbox_xywh = self._parse_output(mask_np)

        frame = frame.copy()

        if mask_visualization_path is not None:
            visualize_mask(frame, mask_np*255, mask_visualization_path)
        if bbox_visualization_path is not None:
            visualize_bbox(frame, bbox_xywh, bbox_visualization_path)

        # del frame_tensor, output_prob
        torch.cuda.empty_cache()

        return bbox_xywh

    def _parse_output(
            self, 
            mask_np: np.ndarray,
    ) -> List[int]:
        # Perform erosion on the mask
        kernel = np.ones((self.erosion_size, self.erosion_size), np.uint8)
        mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1)

        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            bbox = [-1, -1, 0, 0]

        return bbox
