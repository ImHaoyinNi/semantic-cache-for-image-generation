# Create callback for intermediate images
from pathlib import Path

import torch


def create_save_images_callback(save_dir: str = "intermediate_images", save_every: int = 5):
    """Create a callback function for saving intermediate images"""
    def save_images_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if step_index % save_every == 0:
            latents = callback_kwargs["latents"]
            with torch.no_grad():
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image = pipe.image_processor.postprocess(image, output_type="pil")[0]

                intermediate_dir = Path(save_dir)
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                image.save(intermediate_dir / f"step_{step_index:03d}.png")
        return callback_kwargs
    return save_images_on_step_end


def create_save_latents_callback(save_dir: str = "intermediate_latents", save_every: int = 5):
    """Create a callback function for saving intermediate latents"""
    def save_latent_on_step_end(pipe, step_index, timestep, callback_kwargs):
        if step_index % save_every == 0:
            latents = callback_kwargs["latents"]
            intermediate_dir = Path(save_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            torch.save(latents.cpu(), intermediate_dir / f"step_{step_index:03d}.pt")
        return callback_kwargs
    return save_latent_on_step_end


# Backward compatibility - keep old function names
def save_images_on_step_end(pipe, step_index, timestep, callback_kwargs):
    return create_save_images_callback()(pipe, step_index, timestep, callback_kwargs)


def save_latent_on_step_end(pipe, step_index, timestep, callback_kwargs):
    return create_save_latents_callback()(pipe, step_index, timestep, callback_kwargs)