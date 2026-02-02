# SDXL + ControlNet + Refiner
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel
)

def load_sdxl(model_id, controlnet_id, use_refiner):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    refiner = None
    if use_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

    return pipe, refiner
