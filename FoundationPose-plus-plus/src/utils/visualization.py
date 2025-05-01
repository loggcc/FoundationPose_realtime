import cv2
import numpy as np
from PIL import Image
from typing import List


def visualize_mask(
        image: np.ndarray, 
        mask: np.ndarray,
        save_path: str
):
    # Ensure mask is in 'RGBA' mode
    mask_rgba = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')

    # Create an alpha mask where the non-zero regions are semi-transparent
    alpha = 128  # Semi-transparent
    mask_data = mask_rgba.getdata()
    new_data = []
    for item in mask_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # Background pixel, make it fully transparent
            new_data.append((0, 0, 0, 0))
        else:
            # Mask pixel, set desired transparency
            new_data.append((item[0], item[1], item[2], alpha))
    mask_rgba.putdata(new_data)

    # Convert the original image to 'RGBA'
    image_rgba = Image.fromarray(image.astype(np.uint8)).convert('RGBA')

    # Overlay the mask onto the image
    overlaid_image = Image.alpha_composite(image_rgba, mask_rgba)

    # Convert back to 'RGB' mode if you don't need transparency in the saved image
    overlaid_image = overlaid_image.convert('RGB')

    overlaid_image.save(save_path)


def visualize_bbox(
        image: np.ndarray, 
        bbox: List[int], 
        save_path: str
):
    """
    Visualize the bounding box on the image and save it to the result path.
    """
    if image is None:
        return  # If the image can't be read, skip it

    if bbox is None or bbox[0] == -1:
        return  # Skip invalid bounding boxes

    # Unpack the bounding box coordinates (x, y, w, h)
    x, y, w, h = bbox
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))

    # Draw the bounding box on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Save the image to the visualization path
    cv2.imwrite(save_path, image)