import gradio as gr
from omni_zero import OmniZeroSingle

def generate(
    seed=42,
    prompt="A person",
    negative_prompt="blurry, out of focus",
    guidance_scale=3.0,
    number_of_images=1,
    number_of_steps=10,
    base_image="https://github.com/okaris/omni-zero/assets/1448702/2ca63443-c7f3-4ba6-95c1-2a341414865f",
    base_image_strength=0.15,
    composition_image="https://github.com/okaris/omni-zero/assets/1448702/2ca63443-c7f3-4ba6-95c1-2a341414865f",
    composition_image_strength=1.0,
    style_image="https://github.com/okaris/omni-zero/assets/1448702/64dc150b-f683-41b1-be23-b6a52c771584",
    style_image_strength=1.0,
    identity_image="https://github.com/okaris/omni-zero/assets/1448702/ba193a3a-f90e-4461-848a-560454531c58",
    identity_image_strength=1.0,
    depth_image=None,
    depth_image_strength=0.5,
):
    
    omni_zero = OmniZeroSingle(
        base_model="frankjoshua/albedobaseXL_v13",
    )

    images = omni_zero.generate(
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

    # for i, image in enumerate(images):
    #     image.save(f"oz_output_{i}.jpg")
    return images

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", value="A person")
            with gr.Row():
                negative_prompt = gr.Textbox(label="Negative Prompt", value="blurry, out of focus")
            with gr.Row():
                seed = gr.Slider(label="Seed",step=1, minimum=0, maximum=10000000, value=42)
                number_of_images = gr.Slider(label="Number of Outputs",step=1, minimum=1, maximum=4, value=1)
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance Scale",step=0.1, minimum=0.0, maximum=14.0, value=3.0)
                number_of_steps = gr.Slider(label="Number of Steps",step=1, minimum=1, maximum=50, value=10)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        base_image = gr.Image(label="Base Image", value="https://github.com/okaris/omni-zero/assets/1448702/2ca63443-c7f3-4ba6-95c1-2a341414865f")
                    with gr.Row():
                        base_image_strength = gr.Slider(label="Base Image Strength",step=0.01, minimum=0.0, maximum=1.0, value=0.15)
                with gr.Column():
                    with gr.Row():
                        composition_image = gr.Image(label="Composition", value="https://github.com/okaris/omni-zero/assets/1448702/2ca63443-c7f3-4ba6-95c1-2a341414865f")
                    with gr.Row():
                        composition_image_strength = gr.Slider(label="Composition Image Strength",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
            # with gr.Row():
                with gr.Column():
                    with gr.Row():
                        style_image = gr.Image(label="Style Image", value="https://github.com/okaris/omni-zero/assets/1448702/64dc150b-f683-41b1-be23-b6a52c771584")
                    with gr.Row():
                        style_image_strength = gr.Slider(label="Style Image Strength",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
                with gr.Column():
                    with gr.Row():
                        identity_image = gr.Image(label="Identity Image", value="https://github.com/okaris/omni-zero/assets/1448702/ba193a3a-f90e-4461-848a-560454531c58")
                    with gr.Row():
                        identity_image_strength = gr.Slider(label="Identitiy Image Strenght",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
                # with gr.Column():
                #     with gr.Row():
                #         depth_image = gr.Image(label="depth_image", value=None)
                #     with gr.Row():
                #         depth_image_strength = gr.Slider(label="depth_image_strength",step=0.01, minimum=0.0, maximum=1.0, value=0.5)
        with gr.Column():
            with gr.Row():
                out = gr.Image(label="Output(s)", value=None)
            with gr.Row():
                # clear = gr.Button("Clear")
                submit = gr.Button("Generate")
        
                submit.click(generate, inputs=[
                    seed,
                    prompt,
                    negative_prompt,
                    guidance_scale,
                    number_of_images,
                    number_of_steps,
                    base_image,
                    base_image_strength,
                    composition_image,
                    composition_image_strength,
                    style_image,
                    style_image_strength,
                    identity_image,
                    identity_image_strength,
                    ],
                    outputs=[out]
                )
        # clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()