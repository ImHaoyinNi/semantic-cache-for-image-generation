"""
SDXL Pipeline for local image generation
"""
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
from PIL import Image

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, EulerDiscreteScheduler
from pathlib import Path

from src.sdxl_pipeline_utils import create_save_images_callback, create_save_latents_callback
from src.utils import get_root_path

logger = logging.getLogger(__name__)


class EulerAncestralDiscreteScheduler:
    pass


class SDXLWorkflow:
    """SDXL Image Generator"""
    def __init__(self,
                 model_path: Path,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        """
        Initialize SDXL workflow

        Args:
            model_path: Path to SDXL checkpoint file
            lora_directory: Directory containing LoRA files
            device: Device to run inference on
            dtype: Data type for inference
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.img2img_pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None

        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def generate_image(self,
                       prompt: str,
                       negative_prompt: str = "",
                       width: int = 1024,
                       height: int = 1024,
                       num_inference_steps: int = 20,
                       guidance_scale: float = 7.5,
                       seed: Optional[int] = None,
                       save_intermediate: bool = False,
                       intermediate_path: Path = None) -> Optional[Image.Image]:
        """
        Generate an image using the loaded workflow

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            seed: Random seed for reproducibility
            save_intermediate:
            intermediate_path: Path to save intermediate image

        Returns:
            Generated PIL Image or None if failed
        """
        if not self.pipeline:
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            # CUDA
            self.pipeline = self.pipeline.to(self.device)
        try:
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
            callback_on_step_end = None
            if save_intermediate:
                if intermediate_path is None:
                    intermediate_path = Path(get_root_path()) / "output/intermediate"
                # callback_on_step_end = create_save_images_callback(
                #     save_dir=str(intermediate_path),
                #     save_every=5
                # )
                callback_on_step_end = create_save_latents_callback(
                    save_dir=str(intermediate_path),
                    save_every=5
                )

            # Generate image
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=callback_on_step_end,
                return_dict=True
            )

            image = result.images[0]
            logger.info("Image generated successfully")
            return image
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return None

    def generate_image_from_latents(self,
                                    prompt: str,
                                    latents: torch.Tensor,
                                    start_step: int = 0,
                                    negative_prompt: str = "",
                                    num_inference_steps: int = 20,
                                    guidance_scale: float = 7.5,
                                    save_intermediate: bool = False,
                                    intermediate_path: Path = None) -> Optional[Image.Image]:
        """Continue generation from saved latents"""
        if not self.pipeline:
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline = self.pipeline.to(self.device)

        try:
            logger.info(f"Continuing generation from step {start_step}...")

            # Set up scheduler for remaining steps
            self.pipeline.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.pipeline.scheduler.timesteps

            # Skip already completed steps
            if start_step > 0:
                timesteps = timesteps[start_step:]

            callback_on_step_end = None
            if save_intermediate:
                if intermediate_path is None:
                    intermediate_path = Path(get_root_path()) / "output/intermediate_images"
                callback_on_step_end = create_save_images_callback(
                    save_dir=str(intermediate_path),
                    save_every=1
                )

            # Continue denoising from provided latents
            latents = latents.to(self.pipeline.device)

            # Get text embeddings using the pipeline's encode_prompt method
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                self.pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device=self.pipeline.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=guidance_scale > 1.0,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=None,
                )
            )

            # Prepare for classifier-free guidance
            if guidance_scale > 1.0:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            # Use the pipeline's denoising loop but manually control it
            # This is a simplified approach that should work better
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    # Expand latents for classifier-free guidance
                    latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                    latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                    # Get proper dimensions from latents
                    height = latents.shape[2] * 8  # VAE downscale factor
                    width = latents.shape[3] * 8

                    # Create time_ids for SDXL
                    time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=prompt_embeds.dtype, device=self.pipeline.device)
                    if guidance_scale > 1.0:
                        time_ids = time_ids.repeat(2, 1)

                    added_cond_kwargs = {
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids
                    }

                    # Predict noise
                    noise_pred = self.pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    # Classifier-free guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Compute previous latents
                    latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    # Call callback if provided
                    if callback_on_step_end:
                        callback_kwargs = {"latents": latents}
                        current_step = start_step + i
                        callback_on_step_end(self.pipeline, current_step, t, callback_kwargs)

            # Decode final latents
            with torch.no_grad():
                image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
                image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            logger.info("Image generated successfully from latents")
            return image

        except Exception as e:
            logger.error(f"Failed to generate from latents: {e}")
            return None

    def generate_image_from_image(self,
                                  prompt: str,
                                  init_image: Image.Image,
                                  negative_prompt: str = "",
                                  strength: float = 1,
                                  num_inference_steps: int = 20,
                                  guidance_scale: float = 7.5,
                                  save_intermediate: bool = False,
                                  intermediate_path: Path = None) -> Optional[Image.Image]:
        """
        Generate an image using an existing image as starting point

        Args:
            prompt: Text prompt for image generation
            init_image: PIL Image to use as starting point
            negative_prompt: Negative prompt
            strength: How much to transform the image (0.0-1.0, higher = more change)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            save_intermediate: Whether to save intermediate steps
            intermediate_path: Path to save intermediate steps

        Returns:
            Generated PIL Image or None if failed
        """
        if not self.img2img_pipeline:
            if not self.pipeline:
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
                self.pipeline = self.pipeline.to(self.device)

            self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                text_encoder_2=self.pipeline.text_encoder_2,
                tokenizer=self.pipeline.tokenizer,
                tokenizer_2=self.pipeline.tokenizer_2,
                unet=self.pipeline.unet,
                scheduler=self.pipeline.scheduler,
            )
        try:
            logger.info(f"Generating image from image with prompt: {prompt[:50]}...")
            callback_on_step_end = None
            if save_intermediate:
                if intermediate_path is None:
                    intermediate_path = Path(get_root_path()) / "output/intermediate_images"
                callback_on_step_end = create_save_images_callback(
                    save_dir=str(intermediate_path),
                    save_every=5
                )
            # Generate image
            result = self.img2img_pipeline(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=callback_on_step_end,
                return_dict=True
            )
            image = result.images[0]
            logger.info("Image generated successfully from input image")
            return image
        except Exception as e:
            logger.error(f"Failed to generate image from image: {e}")
            return None

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the loaded workflow"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "pipeline_loaded": self.pipeline is not None
        }

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Pipeline cleanup completed")
