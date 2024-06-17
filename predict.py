# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from omni_zero import OmniZeroSingle

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.omni_zero = OmniZeroSingle(
            base_model="frankjoshua/albedobaseXL_v13",
        )

    def predict(
        self,
        seed: int = Input(description="Random seed for the model", default=42),
        prompt: str = Input(description="Prompt for the model", default="A person"),
        negative_prompt: str = Input(description="Negative prompt for the model", default="blurry, out of focus"),
        guidance_scale: float = Input(description="Guidance scale for the model", default=3.0, ge=0.0, le=14.0),
        number_of_images: int = Input(description="Number of images to generate", default=1, ge=1, le=4),
        number_of_steps: int = Input(description="Number of steps for the model", default=10, ge=1, le=50),
        base_image: Path = Input(description="Base image for the model"),
        base_image_strength: float = Input(description="Base image strength for the model", default=0.15, ge=0.0, le=1.0),
        composition_image: Path = Input(description="Composition image for the model"),
        composition_image_strength: float = Input(description="Composition image strength for the model", default=1.0, ge=0.0, le=1.0),
        style_image: Path = Input(description="Style image for the model"),
        style_image_strength: float = Input(description="Style image strength for the model", default=1.0, ge=0.0, le=1.0),
        identity_image: Path = Input(description="Identity image for the model"),
        identity_image_strength: float = Input(description="Identity image strength for the model", default=1.0, ge=0.0, le=1.0),
        depth_image: Path = Input(description="Depth image for the model"),
        depth_image_strength: float = Input(description="Depth image strength for the model, if not supplied the composition image will be used for depth", default=0.5, ge=0.0, le=1.0),
    ) -> Path:
        """Run a single prediction on the model"""
        images = self.omni_zero.generate(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            number_of_images=number_of_images,
            number_of_steps=number_of_steps,
            base_image=base_image,
            base_image_strength=base_image_strength,
            composition_image=composition_image,
            composition_image_strength=composition_image_strength,
            style_image=style_image,
            style_image_strength=style_image_strength,
            identity_image=identity_image,
            identity_image_strength=identity_image_strength,
            depth_image=depth_image,
            depth_image_strength=depth_image_strength,
        )
        
        outputs = []
        for i, image in enumerate(images):
            output_path = f"oz_output_{i}.jpg"
            image.save(output_path)
            outputs.append(Path(output_path))
        
        return outputs