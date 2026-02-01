# =====================================================
# IMPORTS
# =====================================================
import os
import cv2
import torch
import requests
import cloudinary
import cloudinary.uploader

import numpy as np
from io import BytesIO
from PIL import Image

from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel
)


# =====================================================
# SÉCURISATION DES VARIABLES D’ENV
# =====================================================
def get_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"❌ Variable d’environnement manquante : {name}")
    return value


# =====================================================
# CONFIGURATION CLOUDINARY
# =====================================================
cloudinary.config(
    cloud_name=get_env("CLOUDINARY_CLOUD_NAME"),
    api_key=get_env("CLOUDINARY_API_KEY"),
    api_secret=get_env("CLOUDINARY_API_SECRET"),
    secure=True
)

print("✅ Cloudinary configuré")


# =====================================================
# MODÈLES
# =====================================================
SDXL_MODEL = "SG161222/RealVisXL_V4.0"
CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    SDXL_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

print("✅ SDXL + ControlNet chargé")


# =====================================================
# FONCTION : LOAD IMAGE DEPUIS URL
# =====================================================
def load_image_from_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


# =====================================================
# FONCTION : CANNY EDGE (CONTROL IMAGE)
# =====================================================
def make_canny(image: Image.Image, low=100, high=200):
    img = np.array(image)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges)


# =====================================================
# IMAGE D’ENTRÉE (URL CLOUDINARY)
# =====================================================
INPUT_IMAGE_URL = (
    "https://res.cloudinary.com/ddmzn1508/image/upload/"
    "v1769938551/BAC_CHAMBRE_wd3mo8.jpg"
)

init_image = load_image_from_url(INPUT_IMAGE_URL)
control_image = make_canny(init_image)

print("📥 Image + ControlNet Canny prêts")


# =====================================================
# PROMPT — CRÉATIF MAIS CONTRÔLÉ
# =====================================================
prompt = (
    "Photographie d’intérieur réaliste d’une chambre contemporaine haut de gamme, "
    "ambiance chaleureuse et lumineuse, "
    "lumière naturelle directionnelle, "
    "meilleurs matériaux, bois naturel, textile premium, "
    "volumes mieux lisibles, "
    "contraste équilibré, "
    "photographie immobilière professionnelle, "
    "ultra realistic, high detail, sharp focus"
)

negative_prompt = (
    "cartoon, illustration, anime, painting, "
    "3d render, cgi, unreal engine look, "
    "distorted geometry, warped walls, "
    "broken perspective, "
    "fisheye, extreme distortion, "
    "overexposed, underexposed, "
    "people, text, logo, watermark"
)


# =====================================================
# GÉNÉRATION SDXL + CONTROLNET
# =====================================================
generator = torch.Generator("cuda").manual_seed(123456)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=control_image,
    strength=0.60,                 # ⭐ créatif MAIS stable
    guidance_scale=6.5,
    num_inference_steps=40,
    width=1024,
    height=1024,
    generator=generator
).images[0]


# =====================================================
# SAUVEGARDE LOCALE
# =====================================================
OUTPUT_PATH = "sdxl_controlnet_chambre.png"
image.save(OUTPUT_PATH)

print("💾 Image générée avec ControlNet")


# =====================================================
# UPLOAD CLOUDINARY
# =====================================================
result = cloudinary.uploader.upload(
    OUTPUT_PATH,
    folder="sdxl_outputs/controlnet",
    public_id="BAC_CHAMBRE_controlnet",
    overwrite=True
)

print("✅ Upload terminé")
print("🌐 URL :", result["secure_url"])
