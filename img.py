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
# CONFIGURATION CLOUDINARY (ENV VARS)
# =====================================================
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

print("✅ Cloudinary configuré")


# =====================================================
# MODÈLE SDXL (IMG2IMG RÉALISTE)
# =====================================================
MODEL_ID = "SG161222/RealVisXL_V4.0"


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optimisations GPU
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

print("✅ SDXL Img2Img chargé")


# =====================================================
# FONCTION : LOAD IMAGE DEPUIS URL
# =====================================================
def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


# =====================================================
# UPLOAD IMAGE SOURCE (INPUT)
# =====================================================
INPUT_IMAGE_PATH = "BAC_CHAMBRE.jpg"   # image locale à améliorer

input_upload = cloudinary.uploader.upload(
    INPUT_IMAGE_PATH,
    folder="sdxl_inputs",
    public_id="building_01",
    overwrite=True
)

input_url = input_upload["secure_url"]

print("📥 Image source uploadée :", input_url)


# =====================================================
# CHARGEMENT IMAGE INIT
# =====================================================
init_image = load_image_from_url(input_url)


# =====================================================
# PROMPT ARCHITECTURAL RÉALISTE
# =====================================================
prompt = (
    "Photographie architecturale réaliste d’un bâtiment contemporain, "
    "architecture moderne haut de gamme, lignes épurées, "
    "volumes clairs et bien proportionnés, "
    "façade en béton brut, verre clair et métal, "
    "détails constructifs précis, joints visibles, "
    "vue en perspective à hauteur d’homme, "
    "camera eye level, focal length 24mm, "
    "wide shot, building fully visible, no crop, "
    "éclairage naturel réaliste, lumière douce de fin de journée, "
    "ombres cohérentes, global illumination naturelle, "
    "environnement urbain sobre, végétation réaliste, "
    "photographie d’architecture professionnelle, "
    "ultra realistic, high detail, sharp focus, "
    "physically accurate lighting, real materials"
)

negative_prompt = (
    "cartoon, illustration, anime, painting, "
    "3d render, cgi, unreal engine look, "
    "distorted perspective, warped lines, "
    "broken geometry, floating buildings, "
    "unrealistic scale, close-up, cropped, "
    "fisheye, extreme distortion, "
    "overexposed, underexposed, flat lighting, "
    "blurry, noise, low detail, "
    "people in foreground, cars too close, "
    "text, logo, watermark"
)


# =====================================================
# GÉNÉRATION IMAGE-TO-IMAGE
# =====================================================
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.30,                # ⭐ idéal archviz (préserve la géométrie)
    guidance_scale=6.0,
    num_inference_steps=35,
    width=1024,
    height=1024
).images[0]


# =====================================================
# SAUVEGARDE LOCALE
# =====================================================
OUTPUT_PATH = "sdxl_img2img_output.png"
image.save(OUTPUT_PATH)

print("💾 Image sauvegardée localement")


# =====================================================
# UPLOAD CLOUDINARY (OUTPUT)
# =====================================================
result = cloudinary.uploader.upload(
    OUTPUT_PATH,
    folder="sdxl_outputs/img2img",
    public_id="building_01_enhanced",
    overwrite=True
)

print("✅ Image améliorée uploadée")
print("🌐 URL :", result["secure_url"])
