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
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel
)
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

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
# CONFIGURATION
# =====================================================
USE_REFINER = True   # True = utilise le refiner SDXL
USE_UPSCALER = True  # True = utilise Real-ESRGAN pour améliorer la qualité
USE_SDXL = True      # True = utilise SDXL, False = upscale seulement


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
# SDXL REFINER (améliore détails, visages, textures)
# =====================================================
if USE_REFINER:
    REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        REFINER_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    refiner.enable_xformers_memory_efficient_attention()
    refiner.enable_vae_slicing()

    print("✅ SDXL Refiner chargé")
else:
    print("⚠️ Refiner désactivé")


# =====================================================
# REAL-ESRGAN (Amélioration qualité / Upscaling)
# =====================================================
if USE_UPSCALER:
    # Modèle RealESRGAN x4
    esrgan_model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, 
        num_block=23, num_grow_ch=32, scale=4
    )
    
    upscaler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=esrgan_model,
        tile=400,           # Traite par tuiles pour économiser VRAM
        tile_pad=10,
        pre_pad=0,
        half=True           # FP16 pour économiser VRAM
    )
    print("✅ Real-ESRGAN (upscaler) chargé")
else:
    print("⚠️ Upscaler désactivé")


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


def get_output_dimensions(image: Image.Image, max_size: int = 1024) -> tuple:
    """
    Calcule les dimensions de sortie en préservant le ratio d'aspect.
    Les dimensions sont arrondies au multiple de 8 (requis par SDXL).
    """
    width, height = image.size
    aspect_ratio = width / height
    
    if width >= height:
        # Image paysage ou carrée
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        # Image portrait
        new_height = max_size
        new_width = int(max_size * aspect_ratio)
    
    # Arrondir au multiple de 8 (requis par SDXL)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # S'assurer d'un minimum de 512
    new_width = max(512, new_width)
    new_height = max(512, new_height)
    
    return new_width, new_height


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


# =====================================================# NEGATIVE PROMPTS PAR TYPE DE SCÈNE
# =====================================================
NEGATIVE_PROMPTS = {
    "INTERIOR": (
        # Qualité générale
        "cartoon, illustration, anime, painting, sketch, 3d render, cgi, "
        "blurry, low quality, noise, artifacts, "
        
        # Spécifique intérieur - STRUCTURE
        "warped walls, curved walls, distorted perspective, "
        "broken geometry, impossible room layout, "
        "wrong ceiling height, disproportionate room, "
        
        # BLOQUER AJOUTS NON DEMANDÉS
        "added objects, new objects, extra objects, "
        "added furniture, extra furniture, new furniture, "
        "added lights, new lights, extra lights, spotlights, ceiling lights, "
        "added decorations, new decorations, extra decorations, "
        "added plants, new plants, extra plants, "
        "added curtains, new curtains, "
        "modified walls, changed walls, different walls, "
        "modified floor, changed floor, different floor, "
        "modified ceiling, changed ceiling, different ceiling, "
        "removed objects, missing objects, "
        
        # Anatomie (si personne demandée)
        "multiple heads, extra limbs, three arms, four arms, "
        "fused fingers, six fingers, missing arms, "
        "no face, faceless, blank face, "
        
        # Autres
        "text, watermark, logo"
    ),
    
    "EXTERIOR": (
        # Qualité générale
        "cartoon, illustration, anime, painting, sketch, 3d render, cgi, "
        "blurry, low quality, noise, artifacts, "
        
        # Spécifique extérieur
        "distorted building, impossible architecture, "
        "warped façade, broken perspective, tilted verticals, "
        "unrealistic scale, giant trees, tiny cars, "
        "floating buildings, disconnected structure, "
        
        # Nature et environnement
        "fake trees, plastic plants, wrong vegetation scale, "
        "unnatural sky, fake clouds, "
        
        # Anatomie (si personne)
        "deformed face, bad anatomy, extra limbs, "
        
        # Changements non désirés
        "added windows, removed floors, changed materials, "
        "extra buildings, modified landscape, "
        
        # Autres
        "text, watermark, logo"
    ),
    
    "AERIAL": (
        # Qualité générale
        "cartoon, illustration, anime, painting, sketch, 3d render, cgi, "
        "blurry, low quality, noise, artifacts, "
        
        # Spécifique aérien
        "distorted perspective, impossible angle, "
        "warped buildings, curved straight lines, "
        "unrealistic scale, wrong proportions, "
        "floating structures, disconnected roads, "
        
        # Environnement urbain
        "fake vegetation, plastic trees, "
        "unnatural patterns, grid distortion, "
        
        # Changements non désirés
        "added buildings, removed structures, "
        "modified urban layout, changed landscape, "
        
        # Autres
        "text, watermark, logo, people, cars"
    )
}


# =====================================================# IMAGE D’ENTRÉE
# =====================================================
INPUT_IMAGE_URL = (
    "https://res.cloudinary.com/ddmzn1508/image/upload/"
    "v1769946149/1272fc67-ede0-4dbb-9d3a-f21f4ec07c79.png"
)

init_image = load_image_from_url(INPUT_IMAGE_URL)
control_image = make_canny(init_image)

# Calculer dimensions de sortie (préserve le ratio)
output_width, output_height = get_output_dimensions(init_image)
print(f"📍 Image d'entrée : {init_image.size[0]}x{init_image.size[1]}")
print(f"📍 Dimensions de sortie : {output_width}x{output_height} (ratio préservé)")

scene_type = detect_scene_type(init_image)
SCENE_PROMPT = SCENE_PROMPTS[scene_type]
SCENE_NEGATIVE_PROMPT = NEGATIVE_PROMPTS[scene_type]

# Message détaillé de confirmation
print("\n" + "="*50)
print(f"🎯 SCÈNE DÉTECTÉE : {scene_type}")
print("="*50)
if scene_type == "INTERIOR":
    print("   📍 Type : Vue intérieure")
    print("   🏠 Optimisé pour : chambres, salons, bureaux...")
elif scene_type == "EXTERIOR":
    print("   📍 Type : Vue extérieure")
    print("   🏢 Optimisé pour : façades, bâtiments, jardins...")
elif scene_type == "AERIAL":
    print("   📍 Type : Vue aérienne")
    print("   🚁 Optimisé pour : vues drone, plans larges...")
print("="*50 + "\n")


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
    "Améliorer la qualité de l'image,je veux une vue aérienne plus réaliste"
)

FINAL_PROMPT = f"{USER_PROMPT}, {BASE_PROMPT}, {SCENE_PROMPT}"


# =====================================================
# NEGATIVE PROMPT FINAL (BASE + SCÈNE)
# =====================================================
# Negative prompt de base commun
BASE_NEGATIVE = (
    "overexposed, underexposed, flat lighting, "
    "compression artifacts, pixelated"
)

FINAL_NEGATIVE_PROMPT = f"{SCENE_NEGATIVE_PROMPT}, {BASE_NEGATIVE}"


# =====================================================
# GÉNÉRATION
# =====================================================
generator = torch.Generator("cuda").manual_seed(123456)

if USE_SDXL:
    # Étape 1 : Génération avec ControlNet
    base_image = pipe(
        prompt=FINAL_PROMPT,
        negative_prompt=FINAL_NEGATIVE_PROMPT,
        image=init_image,
        control_image=control_image,

        strength=0.30,                      # ⬇️ très bas pour amélioration qualité
        controlnet_conditioning_scale=0.80, # ⬆️ très élevé pour garder la structure
        guidance_scale=9.0,
        num_inference_steps=40,

        width=output_width,
        height=output_height,
        generator=generator
    ).images[0]

    if USE_REFINER:
        print("🚧 Étape 1/2 : Image de base générée")
        
        # Étape 2 : Refinement (améliore détails)
        image = refiner(
            prompt=FINAL_PROMPT,
            negative_prompt=FINAL_NEGATIVE_PROMPT,
            image=base_image,
            strength=0.20,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=torch.Generator("cuda").manual_seed(123456)
        ).images[0]
        
        print("✅ Étape 2/2 : Refinement terminé")
    else:
        image = base_image
        print("✅ SDXL terminé")
else:
    # Pas de SDXL, juste l'image originale
    image = init_image
    print("⚠️ SDXL désactivé, passage direct à l'upscaling")


# =====================================================
# UPSCALING (Real-ESRGAN)
# =====================================================
if USE_UPSCALER:
    print("🔍 Upscaling avec Real-ESRGAN...")
    
    # Convertir PIL → numpy pour Real-ESRGAN
    img_np = np.array(image)
    
    # Upscale x4
    upscaled_np, _ = upscaler.enhance(img_np, outscale=4)
    
    # Convertir numpy → PIL
    image = Image.fromarray(upscaled_np)
    
    print(f"✅ Upscaling terminé : {image.size[0]}x{image.size[1]}")


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
