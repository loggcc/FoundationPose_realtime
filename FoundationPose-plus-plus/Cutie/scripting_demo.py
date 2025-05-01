import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model


def overlay_mask_on_image(image, mask):
    """
    Overlays the segmentation mask onto the original image.

    Args:
        image (PIL.Image.Image): Original RGB image.
        mask (PIL.Image.Image): Segmentation mask image with mode 'P'.

    Returns:
        PIL.Image.Image: Image with the mask overlaid.
    """
    # Ensure mask is in 'RGBA' mode
    mask_rgba = mask.convert('RGBA')

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
    image_rgba = image.convert('RGBA')

    # Overlay the mask onto the image
    overlaid_image = Image.alpha_composite(image_rgba, mask_rgba)

    # Convert back to 'RGB' mode if you don't need transparency in the saved image
    overlaid_image = overlaid_image.convert('RGB')

    return overlaid_image


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    # the processor matches the shorter edge of the input to this size
    # you might want to experiment with different sizes, -1 keeps the original size
    processor.max_internal_size = -1

    image_path = '/home/admin01/Data/PsiRobot/dataset-postprocess/test/raw_data_labeled_indiced_v3/episode/133850/camera/d455_2/color'
    # ordering is important
    images = sorted(os.listdir(image_path))
    images.reverse()

    # mask for the first frame
    # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
    # and definitely NOT a colored RGB image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"
    mask = Image.open('/home/admin01/Data/PsiRobot/dataset-postprocess/test/temp_masks/episode/133850/d455_2/1x4_0/init_mask_395.png')
    assert mask.mode in ['L', 'P']
    
    if mask.mode != 'P':
        mask = mask.convert('P')

    # palette is for visualization
    palette = mask.getpalette()

    # the number of objects is determined by counting the unique values in the mask
    # common mistake: if the mask is resized w/ interpolation, there might be new unique values
    objects = np.unique(np.array(mask))
    # background "0" does not count as an object
    objects = objects[objects != 0].tolist()

    mask = torch.from_numpy(np.array(mask)).cuda()

    for ti, image_name in enumerate(images):
        # load the image as RGB; normalization is done within the model
        image = Image.open(os.path.join(image_path, image_name))
        image_tenser = to_tensor(image).cuda().float()

        if ti == 0:
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step(image_tenser, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step(image_tenser)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask(output_prob, segment_threshold=0.1)

        # visualize prediction
        mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
        mask.putpalette(palette)

        save_dir = "/home/admin01/Data/PsiRobot/dataset-postprocess/test/cutie_mask_seq"
        os.makedirs(save_dir, exist_ok=True)

        overlaid_img = overlay_mask_on_image(image.copy(), mask.copy())
        overlaid_img.save(os.path.join(save_dir, f"{ti}_overlaid.png"))
        mask.save(os.path.join(save_dir, f"{ti}_mask.png"))
        # mask.show()


main()