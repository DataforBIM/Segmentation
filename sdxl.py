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
# Prompt (réalisme forcé)
# -------------------------
prompt = (
    "Photographie réaliste d’un chat Siamois adulte, "
    "pelage court crème avec masque brun foncé sur le visage, "
    "les oreilles, les pattes et la queue, "
    "yeux bleus naturels en forme d’amande, "
    "proportions anatomiquement réalistes, "
    "texture du poil très détaillée, "
    "photo DSLR professionnelle, objectif 85mm, "
    "faible profondeur de champ, "
    "éclairage naturel doux, lumière réaliste, "
    "arrière-plan flou, "
    "animal réel, photo animalière, "
    "ultra realistic, high detail, sharp focus"
)

negative_prompt = (
    "cartoon, illustration, anime, 3d render, cgi, "
    "kawaii, cute, chibi, doll, toy, "
    "big eyes, oversized head, "
    "stylized, painting, drawing, "
    "unrealistic proportions, smooth plastic skin"
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
