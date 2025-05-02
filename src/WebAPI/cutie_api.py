import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response
import h5py
import json

import sys
src_path = os.path.join(os.path.dirname(__file__), "..")
cutie_path = os.path.join(src_path, "Cutie")
if cutie_path not in sys.path:
    sys.path.append(cutie_path)

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

class CutieProcessor:
    def __init__(self):
        self.cutie = get_default_model()
        self.cutie_processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.cutie_processor.max_internal_size = -1

    def process_sequence(self, sequence, init_mask_path, seg_threshold, erosion_size):
        # Load initial mask
        init_mask = Image.open(init_mask_path)
        if init_mask.mode not in ['L', 'P']:
            init_mask = init_mask.convert('L')
        mask_np = np.array(init_mask)
        init_mask_tensor = torch.from_numpy(mask_np).cuda()

        # Get list of objects in the mask (excluding background)
        objects = np.unique(mask_np)
        objects = objects[objects != 0].tolist()

        masks_sequence = []

        with torch.no_grad():
            for idx, img_np in enumerate(sequence):
                img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
                image_tensor = to_tensor(img_pil).cuda().float()

                if idx == 0:
                    output_prob = self.cutie_processor.step(image_tensor, init_mask_tensor, objects=objects)
                else:
                    output_prob = self.cutie_processor.step(image_tensor)

                mask = self.cutie_processor.output_prob_to_mask(output_prob, segment_threshold=seg_threshold)
                mask_np = mask.cpu().numpy()

                kernel = np.ones((erosion_size, erosion_size), np.uint8)
                mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1)

                masks_sequence.append(mask_np)

                del image_tensor, output_prob, mask
                torch.cuda.empty_cache()

            self.cutie_processor.clear_memory()
            torch.cuda.empty_cache()

        return np.array(masks_sequence)

app = FastAPI(title="Cutie Web API")

cutie_processor = CutieProcessor()

@app.post("/cutie")
async def process_masks(
    h5_file_path: str = Form(...),
    meta_info_path: str = Form(...),
    init_mask_path: str = Form(...),
    camera_position: str = Form(...),
    output_h5_path: str = Form(...),
    seg_threshold: float = Form(0.1),
    erosion_size: int = Form(5)
):
    try:
        # Read meta info
        with open(meta_info_path, 'r') as file:
            meta_info = json.load(file)

        # Get target camera
        for camera_name, camera_info in meta_info['cameras'].items():
            if camera_info['position'] == camera_position:
                target_camera_name = camera_name
                break
        # start_index = meta_info["cameras"][target_camera_name]["frames"]["start"]
        # end_index = meta_info["cameras"][target_camera_name]["frames"]["end"]

        # Read HDF5 file
        with h5py.File(h5_file_path, 'r') as file:
            sequence = file['observation_data']['camera'][target_camera_name]['rgb'][:]

        # Process sequence
        masks_sequence = cutie_processor.process_sequence(
            # sequence[start_index:end_index + 1],
            sequence[:],
            init_mask_path,
            seg_threshold,
            erosion_size
        )
        with h5py.File(output_h5_path, "w") as h5file:
            h5file.create_dataset("mask", data=masks_sequence)

        return Response()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAM Web API")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
