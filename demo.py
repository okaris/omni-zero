from omni_zero import OmniZeroSingle

def main():
    
    omni_zero = OmniZeroSingle(
        base_model="frankjoshua/albedobaseXL_v13",
    )

    base_image="https://github.com/okaris/omni-zero/assets/1448702/2ca63443-c7f3-4ba6-95c1-2a341414865f"
    composition_image="https://github.com/okaris/omni-zero/assets/1448702/2ca63443-c7f3-4ba6-95c1-2a341414865f"
    style_image="https://github.com/okaris/omni-zero/assets/1448702/64dc150b-f683-41b1-be23-b6a52c771584"
    identity_image="https://github.com/okaris/omni-zero/assets/1448702/ba193a3a-f90e-4461-848a-560454531c58"

    images = omni_zero.generate(
        seed=42,
        prompt="A person",
        negative_prompt="blurry, out of focus",
        guidance_scale=3.0,
        number_of_images=1,
        number_of_steps=10,
        base_image=base_image,
        base_image_strength=0.15,
        composition_image=composition_image,
        composition_image_strength=1.0,
        style_image=style_image,
        style_image_strength=1.0,
        identity_image=identity_image,
        identity_image_strength=1.0,
        depth_image=None,
        depth_image_strength=0.5, 
    )

    for i, image in enumerate(images):
        image.save(f"oz_output_{i}.jpg")

if __name__ == "__main__":
    main()