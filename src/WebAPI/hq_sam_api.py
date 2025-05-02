import torch
import numpy as np
from typing import List
import os
import cv2
import h5py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
import json
import argparse
import uvicorn
from pydantic import BaseModel
from PIL import Image

from segment_anything import SamPredictor
from segment_anything_hq import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


############################################
# Define Sam Models
############################################
class SAMModel:
    def __init__(self, model_type: str, checkpoint_path: str, device: str):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.transform = ResizeLongestSide(self.predictor.model.image_encoder.img_size)
        self.device = device

    def segment(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        if image.shape[2] != 3:
            raise ValueError("The input image should have 3 channels (RGB).")

        original_image_size = image.shape[:2]  # (H, W)
        transformed_image = self.transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image, device=self.device)
        transformed_image = transformed_image.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]

        self.predictor.set_torch_image(transformed_image, original_image_size)

        boxes = np.array([bbox])  # Shape: [1, 4]
        boxes_transformed = self.transform.apply_boxes(boxes, original_image_size)
        boxes_transformed = torch.as_tensor(boxes_transformed, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_transformed,
                multimask_output=False,
                hq_token_only=False,
            )

        mask = masks[0].cpu().numpy()  # Shape: [H, W]
        return mask


############################################
# Release Web API
############################################
app = FastAPI(title="SAM Web API")

# Initialize device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else "cpu"
)

class Message(BaseModel):
    frame_path: str
    bbox_xywh: List[int]
    output_mask_path: str

@app.post("/hq_sam")
async def segment_image(message: Message):
    try:
        frame = Image.open(message.frame_path)
        frame = np.array(frame)
        bbox_xyxy = [
            message.bbox_xywh[0],
            message.bbox_xywh[1],
            message.bbox_xywh[0] + message.bbox_xywh[2],
            message.bbox_xywh[1] + message.bbox_xywh[3]
        ]

        if frame.shape[2] == 3:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("The input image should have 3 channels.")

        # Perform segmentation
        mask = sam_model.segment(image_rgb, bbox_xyxy)

        # Convert mask to PNG format directly
        mask_uint8 = (mask * 255).astype(np.uint8)
        if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
            mask_uint8 = mask_uint8.squeeze(0)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(message.output_mask_path), exist_ok=True)

        # Save the mask directly as an image file
        cv2.imwrite(message.output_mask_path, mask_uint8)

        return Response(content="Mask generated successfully", media_type="text/plain", status_code=200)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAM Web API")
    parser.add_argument("--checkpoint_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/sam-hq/pretrained_checkpoints/sam_hq_vit_h.pth")
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    # Initialize SAM model with the checkpoint path
    sam_model = SAMModel(
        model_type=os.getenv("HQ_SAM_MODEL_TYPE", "vit_h"),
        checkpoint_path=args.checkpoint_path,
        device=device
    )

    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=args.port)