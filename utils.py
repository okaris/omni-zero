import math
import PIL
import cv2
import numpy as np

from diffusers.utils import load_image

def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
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
    # Open the image
    # image = Image.open(image_path)
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

from PIL import Image

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
