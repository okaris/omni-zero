import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys
sys.path.insert(0, './diffusers/src')

import torch
import torch.nn as nn

from huggingface_hub import snapshot_download
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import ControlNetModel

from transformers import CLIPVisionModelWithProjection

from pipeline import OmniZeroPipeline
from insightface.app import FaceAnalysis
from controlnet_aux import ZoeDetector
from utils import draw_kps, load_and_resize_image, align_images

import cv2
import numpy as np

class OmniZeroSingle():
    def __init__(self,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
    ):
        snapshot_download("okaris/antelopev2", local_dir="./models/antelopev2")
        self.face_analysis = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

        dtype = torch.float16

        ip_adapter_plus_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        ).to("cuda")

        zoedepthnet_path = "okaris/zoe-depth-controlnet-xl"
        zoedepthnet = ControlNetModel.from_pretrained(zoedepthnet_path,torch_dtype=dtype).to("cuda")

        identitiynet_path = "okaris/face-controlnet-xl"
        identitynet = ControlNetModel.from_pretrained(identitiynet_path, torch_dtype=dtype).to("cuda")

        self.zoe_depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        self.pipeline = OmniZeroPipeline.from_pretrained(
            base_model,
            controlnet=[identitynet, zoedepthnet],
            torch_dtype=dtype,
            image_encoder=ip_adapter_plus_image_encoder,
        ).to("cuda")

        config = self.pipeline.scheduler.config
        config["timestep_spacing"] = "trailing"
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", final_sigmas_type="zero")

        self.pipeline.load_ip_adapter(["okaris/ip-adapter-instantid", "h94/IP-Adapter", "h94/IP-Adapter"], subfolder=[None, "sdxl_models", "sdxl_models"], weight_name=["ip-adapter-instantid.bin", "ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors"])
    def get_largest_face_embedding_and_kps(self, image, target_image=None):
        face_info = self.face_analysis.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            return None, None
        largest_face = sorted(face_info, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)[0]
        face_embedding = torch.tensor(largest_face['embedding']).to("cuda")
        if target_image is None:
            target_image = image
        zeros = np.zeros((target_image.size[1], target_image.size[0], 3), dtype=np.uint8)
        face_kps_image = draw_kps(zeros, largest_face['kps'])
        return face_embedding, face_kps_image
    
    def generate(self,
        seed=42,
        prompt="A person",
        negative_prompt="blurry, out of focus",
        guidance_scale=3.0,
        number_of_images=1,
        number_of_steps=10,
        base_image=None,
        base_image_strength=0.15,
        composition_image=None,
        composition_image_strength=1.0,
        style_image=None,
        style_image_strength=1.0,
        identity_image=None,
        identity_image_strength=1.0,
        depth_image=None,
        depth_image_strength=0.5,        
    ):
        resolution = 1024

        if base_image is not None:
            base_image = load_and_resize_image(base_image, resolution, resolution)
        else:
            if composition_image is not None:
                base_image = load_and_resize_image(composition_image, resolution, resolution)
            else:
                raise ValueError("You must provide a base image or a composition image")

        if depth_image is None:
            depth_image = self.zoe_depth_detector(base_image, detect_resolution=resolution, image_resolution=resolution)
        else:
            depth_image = load_and_resize_image(depth_image, resolution, resolution)

        base_image, depth_image = align_images(base_image, depth_image)

        if composition_image is not None:
            composition_image = load_and_resize_image(composition_image, resolution, resolution)
        else: 
            composition_image = base_image

        if style_image is not None:
            style_image = load_and_resize_image(style_image, resolution, resolution)
        else:
            raise ValueError("You must provide a style image")
        
        if identity_image is not None:
            identity_image = load_and_resize_image(identity_image, resolution, resolution)
        else:
            raise ValueError("You must provide an identity image")
        
        face_embedding_identity_image, target_kps = self.get_largest_face_embedding_and_kps(identity_image, base_image)
        if face_embedding_identity_image is None:
            raise ValueError("No face found in the identity image, the image might be cropped too tightly or the face is too small")
        
        face_embedding_base_image, face_kps_base_image = self.get_largest_face_embedding_and_kps(base_image)
        if face_embedding_base_image is not None:
            target_kps = face_kps_base_image

        self.pipeline.set_ip_adapter_scale([identity_image_strength,
            {
                "down": { "block_2": [0.0, 0.0] },
                "up": { "block_0": [0.0, style_image_strength, 0.0] }
            },
            {
                "down": { "block_2": [0.0, composition_image_strength] },
                "up": { "block_0": [0.0, 0.0, 0.0] }
            }
        ])

        generator = torch.Generator(device="cpu").manual_seed(seed)

        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            ip_adapter_image=[face_embedding_identity_image, style_image, composition_image],
            image=base_image,
            control_image=[target_kps, depth_image],
            controlnet_conditioning_scale=[identity_image_strength, depth_image_strength],
            identity_control_indices=[(0,0)],
            num_inference_steps=number_of_steps, 
            num_images_per_prompt=number_of_images,
            strength=(1-base_image_strength),
            generator=generator,
            seed=seed,
        ).images

        return images