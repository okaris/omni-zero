import math
import PIL
from PIL import Image
import cv2
import numpy as np

from diffusers.utils import load_image

def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    """
    Draw keypoints on an image.

    Args:
    image_pil (PIL.Image): Image on which to draw the keypoints.
    kps (list): List of keypoints to draw.
    color_list (list): List of colors to use for drawing the keypoints.

    Returns:
    PIL.Image: Image with keypoints drawn on it.
    """
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    # w, h = image_pil.size
    # out_img = np.zeros([h, w, 3])
    if type(image_pil) == PIL.Image.Image:
        out_img = np.array(image_pil)
    else:
        out_img = image_pil

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def load_and_resize_image(image_path, max_width, max_height, maintain_aspect_ratio=True):
    """
    Load and resize an image to the specified dimensions.
    
    Args:
    image_path (str): Path to the image file.
    max_width (int): Maximum width of the resized image.
    max_height (int): Maximum height of the resized image.
    maintain_aspect_ratio (bool): Whether to maintain the aspect ratio of the image.
    
    Returns:
    PIL.Image: Resized image.
    """

    # Open the image
    if isinstance(image_path, np.ndarray):
        image_path = Image.fromarray(image_path)

    image = load_image(image_path)

    # Get the current width and height of the image
    current_width, current_height = image.size

    if maintain_aspect_ratio:
        # Calculate the aspect ratio of the image
        aspect_ratio = current_width / current_height

        # Calculate the new dimensions based on the max width and height
        if current_width / max_width > current_height / max_height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
    else:
        # Use the max width and height as the new dimensions
        new_width = max_width
        new_height = max_height

    # Ensure the new dimensions are divisible by 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image


def align_images(image1, image2):
    """
    Resize two images to the same dimensions by cropping the larger image(s) to match the smaller one.

    Args:
    image1 (PIL.Image): First image to be aligned.
    image2 (PIL.Image): Second image to be aligned.

    Returns:
    tuple: A tuple containing two images with the same dimensions.
    """
    # Determine the new size by taking the smaller width and height from both images
    new_width = min(image1.size[0], image2.size[0])
    new_height = min(image1.size[1], image2.size[1])

    # Crop both images if necessary
    if image1.size != (new_width, new_height):
        image1 = image1.crop((0, 0, new_width, new_height))
    if image2.size != (new_width, new_height):
        image2 = image2.crop((0, 0, new_width, new_height))

    return image1, image2

def align_images_2(image1, image2):
    """
    Resize and crop the second image to match the dimensions of the first image by
    scaling to aspect fill and then center cropping the extra parts.

    Args:
    image1 (PIL.Image): First image which will act as the reference for alignment.
    image2 (PIL.Image): Second image to be aligned to the first image's dimensions.

    Returns:
    tuple: A tuple containing the first image and the aligned second image.
    """
    # Get dimensions of the first image
    target_width, target_height = image1.size

    # Calculate the aspect ratio of the second image
    aspect_ratio = image2.width / image2.height

    # Calculate dimensions to aspect fill
    if target_width / target_height > aspect_ratio:
        # The first image is wider relative to its height than the second image
        fill_height = target_height
        fill_width = int(fill_height * aspect_ratio)
    else:
        # The first image is taller relative to its width than the second image
        fill_width = target_width
        fill_height = int(fill_width / aspect_ratio)

    # Resize the second image to fill dimensions
    filled_image = image2.resize((fill_width, fill_height), Image.Resampling.LANCZOS)

    # Calculate top-left corner of crop box to center crop
    left = (fill_width - target_width) / 2
    top = (fill_height - target_height) / 2
    right = left + target_width
    bottom = top + target_height

    # Crop the filled image to match the size of the first image
    cropped_image = filled_image.crop((int(left), int(top), int(right), int(bottom)))

    return image1, cropped_image
