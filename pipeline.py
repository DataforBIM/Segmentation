# Orchestration centrale
import torch
from config.settings import *
from models.blip import detect_scene_type
from models.sdxl import load_sdxl
from steps.preprocess import make_canny, compute_output_size
from prompts.base import BASE_PROMPT, BASE_NEGATIVE
from prompts.scenes import SCENE_PROMPTS, NEGATIVE_PROMPTS

def run_pipeline(image, user_prompt):
    scene = detect_scene_type(image)

    prompt = f"{user_prompt}, {BASE_PROMPT}, {SCENE_PROMPTS[scene]}"
    negative = f"{NEGATIVE_PROMPTS[scene]}, {BASE_NEGATIVE}"

    control = make_canny(image)
    width, height = compute_output_size(image, MAX_SIZE)

    pipe, refiner = load_sdxl(
        SDXL_MODEL, CONTROLNET_MODEL, USE_REFINER
    )

    gen = torch.Generator("cuda").manual_seed(SEED)

    base = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=image,
        control_image=control,
        strength=0.30,
        controlnet_conditioning_scale=0.80,
        guidance_scale=9.0,
        num_inference_steps=40,
        width=width,
        height=height,
        generator=gen
    ).images[0]

    if refiner:
        base = refiner(
            prompt=prompt,
            negative_prompt=negative,
            image=base,
            strength=0.20,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=gen
        ).images[0]

    return base
