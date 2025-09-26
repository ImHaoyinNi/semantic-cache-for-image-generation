import logging
import torch
from pathlib import Path
from PIL import Image
from src.sdxl_pipeline import SDXLWorkflow
from src.utils import get_root_path

logging.basicConfig(level=logging.INFO)


class ImageGenerationExperiments:
    def __init__(self, test_id: int):
        self.test_id = test_id
        # Model path
        model_path = get_root_path() / "models/checkpoints/illustriousRealismBy_v10VAE.safetensors"
        if not model_path.exists():
            print(f"ERROR: Model file not found at {model_path}")
            raise SystemExit
        # Initialize generator
        self.generator = SDXLWorkflow(
            model_path=model_path,
            device="cuda"
        )
        # Prompts
        self.pos_prompt = """masterpiece, best quality, high quality, ultra-detailed,
            1girl, black hair, short hair, blue eyes, sexy pose, dynamic pose,
            blue shirt, short sleeve, white mini skirt,
            smiling, looking at viewer, sitting on desk, hold arms,
            bust shot, office, desk,"""
        self.pos_prompt_2 = """masterpiece, best quality, high quality, ultra-detailed,
            1girl, blonde hair, long hair, black eyes, sexy pose, dynamic pose,
            red shirt, black mini skirt,
            crying, looking at viewer, sitting on desk, hold arms,
            bust shot, office, desk,"""
        self.negative_prompt = """multiple views, worst quality, low quality, sketch, error, bad anatomy, bad hands, watermark, ugly, distorted, censored, signature, 3D, logo, extra fingers, extra limbs, text,"""

    def text_to_image(self, test_prompt: str):
        try:
            image = self.generator.generate_image(
                prompt=test_prompt,
                negative_prompt=self.negative_prompt,
                width=1024,
                height=1024,
                num_inference_steps=20,
                guidance_scale=7.5,
                seed=0,
                save_intermediate=True,
                intermediate_path= get_root_path() / "output" / f"text_to_image_{self.test_id}" / "intermediate"
            )
            if image:
                output_path = get_root_path() / "output" / f"text_to_image_{self.test_id}"/ "final_output.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
                print(f"Image saved to: {output_path.absolute()}")
            else:
                print("Failed to generate image")
            self.generator.cleanup()
            print("Test completed successfully!")
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()

    def image_to_image(self, test_prompt: str, init_image: Path, num_inference_steps: int):
        try:
            image = self.generator.generate_image_from_image(
                init_image=Image.open(init_image),
                strength=0.8,
                prompt=test_prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=5.5,
                save_intermediate=True,
                intermediate_path=get_root_path() / "output" / f"image_to_image_{self.test_id}" / "intermediate"
            )
            if image:
                output_path = get_root_path() / "output" / f"image_to_image_{self.test_id}" / "final_output.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
                print(f"Image saved to: {output_path.absolute()}")
            else:
                print("Failed to generate image")
            self.generator.cleanup()
            print("Test completed successfully!")
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()

    def latent_to_image(self, test_prompt: str, latent_path: Path, start_step: int):
        try:
            if not latent_path.exists():
                print(f"ERROR: Latent file not found at {latent_path}")
                return
            latents = torch.load(latent_path, weights_only=True)
            print(f"Loaded latents from {latent_path}")
            image = self.generator.generate_image_from_latents(
                prompt=test_prompt,
                latents=latents,
                start_step=start_step,
                negative_prompt=self.negative_prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                save_intermediate=True,
                intermediate_path=get_root_path() / "output" / f"latent_to_image_{self.test_id}" / "intermediate"
            )

            if image:
                output_path = get_root_path() / "output" / f"latent_to_image_{self.test_id}" / "final_output.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
                print(f"Image saved to: {output_path.absolute()}")
            else:
                print("Failed to generate image from latents")
            self.generator.cleanup()
            print("Latent to image test completed successfully!")
        except Exception as e:
            print(f"Error during latent to image test: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    experiments = ImageGenerationExperiments(test_id=1)
    # experiments.text_to_image(experiments.pos_prompt)
    # init_image_path = get_root_path() / "output/text_to_image_1/final_output.png"
    # experiments.image_to_image(experiments.pos_prompt_2, init_image=init_image_path, num_inference_steps=10)
    latent_path = get_root_path() / "output/text_to_image_1/intermediate/step_05.pt"
    experiments.latent_to_image(experiments.pos_prompt, latent_path=latent_path, start_step=5)
