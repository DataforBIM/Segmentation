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

# 🔧 MODIF : on n'utilise PLUS pipeline()
from transformers import BlipProcessor, BlipForConditionalGeneration


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
# MODÈLES SDXL + CONTROLNET
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
# 🔧 MODIF MAJEURE : MODÈLE VISION BLIP (CAPTIONING)
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

print("✅ Modèle BLIP (scene detection) chargé")


# =====================================================
# UTILITAIRES IMAGE
# =====================================================
def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def make_canny(image: Image.Image, low=80, high=160) -> Image.Image:
    img = np.array(image)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


# =====================================================
# 🔧 MODIF : DÉTECTION SCÈNE STABLE
# =====================================================
def detect_scene_type(image: Image.Image) -> str:
    inputs = blip_processor(image, return_tensors="pt").to(device)

    output = blip_model.generate(
        **inputs,
        max_new_tokens=40
    )

    caption = blip_processor.decode(
        output[0],
        skip_special_tokens=True
    ).lower()

    print("🧠 Caption IA :", caption)

    if any(w in caption for w in ["aerial", "drone", "top view", "bird"]):
        return "AERIAL"

    if any(w in caption for w in ["room", "interior", "bedroom", "living"]):
        return "INTERIOR"

    return "EXTERIOR"


# =====================================================
# SCENE PROMPTS (FR + EN)
# =====================================================
SCENE_PROMPTS = {
    "INTERIOR": (
        "architecture intérieure contemporaine, "
        "interior architectural photography, "
        "wide shot interior, "
        "camera at eye level, "
        "straight verticals, "
        "realistic room proportions"
    ),
    "EXTERIOR": (
        "architecture contemporaine extérieure, "
        "exterior architectural photography, "
        "wide shot exterior, "
        "building fully visible, "
        "camera at eye level, "
        "straight verticals, "
        "realistic scale and proportions"
    ),
    "AERIAL": (
        "vue aérienne architecturale, "
        "aerial architectural photography, "
        "drone view, "
        "oblique aerial perspective, "
        "large scale context visible"
    )
}


# =====================================================
# IMAGE D’ENTRÉE
# =====================================================
INPUT_IMAGE_URL = (
    "https://res.cloudinary.com/ddmzn1508/image/upload/"
    "v1769938551/BAC_CHAMBRE_wd3mo8.jpg"
)

init_image = load_image_from_url(INPUT_IMAGE_URL)
control_image = make_canny(init_image)

scene_type = detect_scene_type(init_image)
SCENE_PROMPT = SCENE_PROMPTS[scene_type]

print(f"🎯 SCÈNE DÉTECTÉE : {scene_type}")


# =====================================================
# PROMPT LAYERING
# =====================================================
BASE_PROMPT = (
    "Photographie architecturale réaliste haut de gamme, "
    "architecture contemporaine, "
    "volumes clairs et bien proportionnés, "
    "géométrie cohérente et stable, "
    "matériaux crédibles et réalistes, "
    "lumière naturelle physiquement correcte, "
    "ombres cohérentes, "
    "composition architecturale équilibrée, "
    "photographie professionnelle, "
    "ultra realistic, high detail, sharp focus"
)

# 🔧 Astuce : phrase utilisateur en FR + EN = meilleur contrôle
# ⚠️ USER_PROMPT en PREMIER pour maximiser son influence
USER_PROMPT = (
    "changer la couleur des draps vers un bleu ciel doux et apaisant, "
    "change the bed sheets color to a soft sky blue"
)

FINAL_PROMPT = f"{USER_PROMPT}, {BASE_PROMPT}, {SCENE_PROMPT}"


# =====================================================
# NEGATIVE PROMPT
# =====================================================
FINAL_NEGATIVE_PROMPT = (
    "cartoon, illustration, anime, painting, "
    "3d render, cgi, unreal engine look, "
    "plastic materials, low poly, "
    "distorted geometry, warped walls, "
    "broken perspective, impossible architecture, "
    "floating objects, unrealistic scale, "
    "fisheye, extreme wide angle distortion, "
    "overexposed, underexposed, flat lighting, "
    "blurry, noise, artifacts, "
    "people, text, logo, watermark"
)


# =====================================================
# GÉNÉRATION
# =====================================================
generator = torch.Generator("cuda").manual_seed(123456)

image = pipe(
    prompt=FINAL_PROMPT,
    negative_prompt=FINAL_NEGATIVE_PROMPT,
    image=init_image,
    control_image=control_image,

    strength=0.55,                      # ⬆️ augmenté (était 0.40) pour plus de modifications
    controlnet_conditioning_scale=0.45, # ⬇️ réduit (était 0.65) pour plus de liberté créative
    guidance_scale=8.5,                 # ⬆️ augmenté (était 7.0) pour mieux suivre le prompt
    num_inference_steps=40,

    width=1024,
    height=1024,
    generator=generator
).images[0]


# =====================================================
# SAUVEGARDE + UPLOAD
# =====================================================
OUTPUT_PATH = "sdxl_archviz_auto_scene.png"
image.save(OUTPUT_PATH)

result = cloudinary.uploader.upload(
    OUTPUT_PATH,
    folder="sdxl_outputs/auto_scene",
    public_id="archviz_auto_scene",
    overwrite=True
)

print("✅ Image générée et uploadée")
print("🌐 URL :", result["secure_url"])
