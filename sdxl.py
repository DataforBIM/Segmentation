import os
import torch
import cloudinary
import cloudinary.uploader
from diffusers import StableDiffusionXLPipeline

# -------------------------
# Cloudinary config (env)
# -------------------------
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

# -------------------------
# Modèle SDXL réaliste (TOP)
# -------------------------
MODEL_ID = "SG161222/RealVisXL_V4.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optimisations A100
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

print("✅ SDXL RealVis chargé")

# -------------------------
# PROMPT – Architecture réaliste
# -------------------------
prompt = (
    "Photographie architecturale réaliste d’un quartier urbain contemporain, "
    "bâtiments à l’échelle réelle, géométrie cohérente, "
    "façades en béton, verre et métal, "
    "voiries réalistes, trottoirs, arbres, mobilier urbain, "
    "organisation urbaine crédible, "
    "photo professionnelle, caméra plein format, "
    "objectif 35mm, perspective réaliste, "
    "lumière naturelle, ombres réalistes, "
    "rendu photographique ultra réaliste, "
    "high detail, sharp focus"
)

negative_prompt = (
    "illustration, concept art, cgi, render, cartoon, "
    "fantasy city, sci-fi, futuristic, "
    "warped perspective, distorted geometry, "
    "floating buildings, impossible structures, "
    "toy city, low detail, blurry"
)

# Seed reproductible
generator = torch.Generator(device="cuda").manual_seed(1234)

# -------------------------
# Génération
# -------------------------
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=7.5,       # 🔥 SDXL aime 6–8
    num_inference_steps=40,   # qualité ++
    height=1024,
    width=1024,
    generator=generator
).images[0]

# -------------------------
# Sauvegarde locale
# -------------------------
local_path = "sdxl_architecture.png"
image.save(local_path)

# -------------------------
# Upload Cloudinary
# -------------------------
result = cloudinary.uploader.upload(
    local_path,
    folder="sdxl_outputs",
    public_id="sdxl_architecture_realistic",
    overwrite=True
)

print("✅ Image uploadée sur Cloudinary")
print("🌐 URL :", result["secure_url"])
