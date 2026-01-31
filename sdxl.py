import os
import torch
import cloudinary
import cloudinary.uploader
from diffusers import StableDiffusionXLPipeline

# =====================================================
# Cloudinary config (variables d’environnement)
# =====================================================
cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
    secure=True
)

# =====================================================
# Chargement du modèle SDXL réaliste
# =====================================================
MODEL_ID = "SG161222/RealVisXL_V4.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optimisations GPU A100
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

print("✅ SDXL RealVis XL chargé avec succès")

# =====================================================
# Prompt – CHAT SIAMOIS PLEIN CORPS (FULL BODY)
# =====================================================
prompt = (
    "Photographie réaliste d’un chat Siamois adulte, "
    "pelage court crème avec masque brun foncé sur le visage, "
    "les oreilles, les pattes et la queue, "
    "yeux bleus naturels en forme d’amande, "
    "proportions anatomiquement réalistes, "
    "texture du poil très détaillée, "

    "full body shot, entire animal visible, "
    "wide shot, camera pulled back, "
    "standing on the ground, "
    "subject centered, correct framing, "
    "no crop, no close-up, "

    "natural lighting, realistic shadows, "
    "background softly blurred but environment visible, "
    "real animal photography, "
    "ultra realistic, high detail"
)

# =====================================================
# Negative prompt – INTERDIRE LE PORTRAIT
# =====================================================
negative_prompt = (
    "close-up, portrait, head shot, face only, cropped, "
    "zoomed in, extreme close-up, "

    "cartoon, illustration, anime, 3d render, cgi, "
    "kawaii, cute, chibi, doll, toy, "
    "big eyes, oversized head, "
    "stylized, painting, drawing, "
    "unrealistic proportions, smooth plastic skin, "
    "blurry, low detail"
)

# =====================================================
# Génération (sans seed → variations naturelles)
# =====================================================
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=6.0,        # 🔑 idéal pour cadrage plein corps
    num_inference_steps=30,    # équilibre qualité / liberté
    height=1024,
    width=1024
).images[0]

# =====================================================
# Sauvegarde locale
# =====================================================
local_path = "sdxl_cat_full_body.png"
image.save(local_path)

# =====================================================
# Upload Cloudinary
# =====================================================
result = cloudinary.uploader.upload(
    local_path,
    folder="sdxl_outputs",
    public_id="sdxl_siamese_full_body",
    overwrite=True
)

print("✅ Image uploadée sur Cloudinary")
print("🌐 URL :", result["secure_url"])
