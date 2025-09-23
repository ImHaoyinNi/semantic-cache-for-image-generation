import logging
import torch
from pathlib import Path
from PIL import Image
from src.sdxl_pipeline import SDXLWorkflow
from src.utils import get_root_path

logging.basicConfig(level=logging.INFO)

# Model path
model_path = get_root_path() / "models/checkpoints/waiNSFWIllustrious_v130.safetensors"
if not model_path.exists():
    print(f"ERROR: Model file not found at {model_path}")
    raise SystemExit

# Initialize generator
generator = SDXLWorkflow(
    model_path=model_path,
    device="cuda"
)

# Prompts
test_prompt = """masterpiece, best quality, high quality, ultra-detailed,
1girl, black hair, short hair, black eyes, sexy pose, dynamic pose,
white shirt, short sleeve, black mini skirt, 
smiling, looking at viewer, sitting on desk, hold arms,  
bust shot, office, desk,"""

negative_prompt = """multiple views, worst quality, low quality, sketch, error, bad anatomy, bad hands, watermark, ugly, distorted, censored, signature, 3D, logo, extra fingers, extra limbs, text,"""


def text_to_image():
    try:
        image = generator.generate_image(
            prompt=test_prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=0,
            save_intermediate=True
        )
        if image:
            output_path = get_root_path() / "output" / "final_text_to_image_output.png"
            image.save(output_path)
            print(f"Image saved to: {output_path.absolute()}")
        else:
            print("Failed to generate image")
        generator.cleanup()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def image_to_image():
    try:
        image = generator.generate_image_from_image(
            init_image=Image.open(get_root_path()/"output/intermediate_images_0/step_010.png"),
            prompt=test_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=10,
            guidance_scale=7.5,
            save_intermediate=True
        )
        if image:
            output_path = Path("final_image_to_image_output.png")
            image.save(output_path)
            print(f"Image saved to: {output_path.absolute()}")
        else:
            print("Failed to generate image")
        generator.cleanup()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


def latent_to_image():
    try:
        # Load saved latents from step 10
        latent_path = get_root_path() / "output/intermediate/step_010.pt"
        if not latent_path.exists():
            print(f"ERROR: Latent file not found at {latent_path}")
            return

        latents = torch.load(latent_path, weights_only=True)
        print(f"Loaded latents from {latent_path}")

        # Continue generation from step 10
        image = generator.generate_image_from_latents(
            prompt=test_prompt,
            latents=latents,
            start_step=10,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            save_intermediate=True
        )

        if image:
            output_path = get_root_path() / "output" / "final_latent_to_image_output.png"
            image.save(output_path)
            print(f"Image saved to: {output_path.absolute()}")
        else:
            print("Failed to generate image from latents")
        generator.cleanup()
        print("Latent to image test completed successfully!")
    except Exception as e:
        print(f"Error during latent to image test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # text_to_image()
    # image_to_image()
    latent_to_image()