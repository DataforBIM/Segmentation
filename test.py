import os
import torch
import cloudinary
import cloudinary.uploader
from diffusers import FluxPipeline

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
# Modèle FLUX
# -------------------------
MODEL_ID = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"  # 🔹 A100 dédiée → pas besoin de "balanced"
)

# Optimisations mémoire / perf
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Si dispo (souvent OK sur Vast)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

print("✅ FLUX chargé")

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

# Seed fixe pour debug
generator = torch.Generator(device="cuda").manual_seed(42)

# -------------------------
# Génération
# -------------------------
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=3.0,        # 🔹 FLUX aime les valeurs basses
    num_inference_steps=32,    # 🔹 sweet spot
    height=1024,
    width=1024,
    generator=generator
).images[0]

# -------------------------
# Sauvegarde temporaire
# -------------------------
local_path = "flux_output.png"
image.save(local_path)

# -------------------------
# Upload Cloudinary
# -------------------------
result = cloudinary.uploader.upload(
    local_path,
    folder="flux_outputs",
    public_id="flux_siamese_realistic",
    overwrite=True
)

print("✅ Image uploadée sur Cloudinary")
print("🌐 URL :", result["secure_url"])
