import json
import argparse
import requests
from typing import List


def generate_mask(
        frame_path: str,
        bbox_xywh: List,
        output_mask_path: str,
        web_api_url: str,
):
    data = {
        'frame_path': frame_path,
        'bbox_xywh': bbox_xywh,
        "output_mask_path": output_mask_path,
    }

    try:
        response = requests.post(web_api_url, json=data)

        if response.status_code == 200:
            print(f"The mask has been saved at {output_mask_path}")
        else:
            print(f"Error: {response.status_code}, {response.json()}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", default="/home/admin01/Data/FoundationPose++/test/test_case/test_case2/color/0.jpg")
    parser.add_argument("--bbox_xywh", type=json.loads, default="[632, 419, 198, 59]")
    parser.add_argument("--output_mask_path", default="/home/admin01/Data/FoundationPose++/test/test_case/test_case2/mask_visualization_img.png")
    parser.add_argument("--web_api_url", type=str, default="http://localhost:9002/hq_sam")

    args = parser.parse_args()

    generate_mask(
        args.frame_path,
        args.bbox_xywh,
        args.output_mask_path,
        args.web_api_url,   
    )