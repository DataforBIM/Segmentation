# =====================================================
# IMPORTS
# =====================================================
import os
import torch
import cloudinary
import cloudinary.uploader
import requests

from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline


# =====================================================
# SÉCURISATION DES VARIABLES D’ENV
# =====================================================
def get_env(name: str) -> str:
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
# MODÈLE SDXL IMG2IMG (RÉALISTE)
# =====================================================
MODEL_ID = "SG161222/RealVisXL_V4.0"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

print("✅ SDXL Img2Img chargé")


# =====================================================
# FONCTION : CHARGER IMAGE DEPUIS URL
# =====================================================
def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


# =====================================================
# IMAGE D’ENTRÉE (URL CLOUDINARY PUBLIQUE)
# =====================================================
INPUT_IMAGE_URL = (
    "https://res.cloudinary.com/ddmzn1508/image/upload/"
    "v1769938551/BAC_CHAMBRE_wd3mo8.jpg"
)

init_image = load_image_from_url(INPUT_IMAGE_URL)
print("📥 Image source chargée :", init_image.size)


# =====================================================
# PROMPT – INTÉRIEUR / CHAMBRE (PHOTOREALISTE)
# =====================================================
prompt = (
    "Photographie d’intérieur réaliste d’une chambre contemporaine, "
    "architecture intérieure haut de gamme, "
    "volumes propres et bien proportionnés, "
    "murs lisses, matériaux réalistes, "
    "bois, textile, surfaces mates naturelles, "
    "mobilier bien aligné, proportions réalistes, "
    "éclairage naturel doux venant des fenêtres, "
    "ombres cohérentes, balance des blancs naturelle, "
    "photographie immobilière professionnelle, "
    "ultra realistic, high detail, sharp focus, "
    "physically accurate lighting"
)

negative_prompt = (
    "cartoon, illustration, anime, painting, "
    "3d render, cgi, unreal engine look, "
    "distorted perspective, warped lines, "
    "broken geometry, unrealistic scale, "
    "fisheye, extreme wide angle distortion, "
    "overexposed, underexposed, flat lighting, "
    "blurry, noise, low detail, "
    "people, text, logo, watermark"
)


# =====================================================
# GÉNÉRATION IMAGE-TO-IMAGE
# =====================================================
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.28,                 # ⭐ parfait pour améliorer sans détruire
    guidance_scale=6.0,
    num_inference_steps=35,
    width=1024,
    height=1024
).images[0]


# =====================================================
# SAUVEGARDE LOCALE
# =====================================================
OUTPUT_PATH = "sdxl_chambre_enhanced.png"
image.save(OUTPUT_PATH)

print("💾 Image sauvegardée localement")


# =====================================================
# UPLOAD CLOUDINARY (OUTPUT)
# =====================================================
result = cloudinary.uploader.upload(
    OUTPUT_PATH,
    folder="sdxl_outputs/img2img",
    public_id="BAC_CHAMBRE_enhanced",
    overwrite=True
)

print("✅ Image améliorée uploadée")
print("🌐 URL :", result["secure_url"])
