import re
import cv2
import argparse
import requests
from PIL import Image
from typing import Union, List, Tuple
import numpy as np


def visualize_bbox(
        image: np.ndarray, 
        bbox_xywh: Union[List, Tuple], 
        output_path: str
):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise ValueError("The image should be a numpy.ndarray.")

    if not (isinstance(bbox_xywh, (list, tuple)) and len(bbox_xywh) == 4):
        raise ValueError("The bbox should be a list or tuple with four elements (x, y, w, h).")

    x1, y1, w, h = bbox_xywh
    x2, y2 = x1 + w, y1 + h

    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    label = f"[{x1}, {y1}, {w}, {h}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 + text_size[1]

    cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    cv2.imwrite(output_path, image)


def _parse_qwen2_vl_output(
        output_text:str,
        img_H: int, 
        img_W: int,
)->List:
    """
    We assume the bbox in the output_text is in the format of (x, y, x, y)
    """
    pattern = r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    
    matches = re.findall(pattern, output_text)
    
    bbox_xyxy_resized = [list(map(int, match)) for match in matches][0]
    
    bbox_xywh = [
        bbox_xyxy_resized[0] * img_W / 1000,
        bbox_xyxy_resized[1] * img_H / 1000,
        (bbox_xyxy_resized[2] - bbox_xyxy_resized[0]) * img_W / 1000,
        (bbox_xyxy_resized[3] - bbox_xyxy_resized[1]) * img_H / 1000,
    ]
    bbox_xywh = list(map(int, bbox_xywh))

    return bbox_xywh


def request_bbox(
        frame_path: str,
        object_name: str,
        visualize_path: str,
        web_api_url: str,
        reference_img_path: str = None,
):    
    frame = np.array(Image.open(frame_path))
    img_H, img_W = frame.shape[0], frame.shape[1]

    if reference_img_path is not None:
        data = {
            "image_paths" : [reference_img_path, frame_path],
            "text_input" : f"第1张图片里是一个{object_name}的照片，你能在第2图片里找到{object_name}的锚框吗？请用(x,y,x,y)的格式给我锚框",
        }        
    else:
        data = {
            "image_paths" : [frame_path],
            "text_input" : f"你能在图片里找到{object_name}的锚框吗？请用(x,y,x,y)的格式给我锚框",
        }

    response = requests.post(web_api_url,json=data)

    if response.status_code == 200:
        output_text = response.json()['output'][0]
        bbox_xywh = _parse_qwen2_vl_output(output_text, img_H, img_W)
        visualize_bbox(frame, bbox_xywh, visualize_path)
        print(f"{bbox_xywh}")
    else:
        print(f"Error: {response.status_code}, {response.json()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/testcase/color/0.jpg", help="The target frame (.jpg/.png/...) path.")
    parser.add_argument("--object_name", type=str, default="蓝色乐高积木", help="The object description. CHINESE will be better than English, for Qwen models.")
    parser.add_argument("--visualize_path", type=str, default="/workspace/yanwenhao/detection/FoundationPose++/visualize/0_v.jpg", help="The visualization image with bbox overlapped on the target frame.")
    parser.add_argument("--web_api_url", type=str, default="http://127.0.0.1:9003/qwen2_vl")
    parser.add_argument("--reference_img_path", type=str, default=None, help="One-shot-learning prompt. With this prompt, the Qwen2-VL performance will be marginally better. \
                        NOTICE: only ONE reference image being provided can peak the best performace.")

    args = parser.parse_args()

    request_bbox(
            args.frame_path,
            args.object_name,
            args.visualize_path,
            args.web_api_url,
            args.reference_img_path,
    )