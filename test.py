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
# MODEL_ID = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="balanced"
)

pipe.enable_attention_slicing()
print("✅ FLUX chargé")

# -------------------------
# Prompt
# -------------------------
prompt = (
    "minimalist apartment floor plan, pure white background, "
    "architectural drawing, top view, clean lines, no shadows"
)

# -------------------------
# Génération
# -------------------------
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_inference_steps=30,
    height=1024,
    width=1024,
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
    public_id="flux_floorplan",
    overwrite=True
)

print("✅ Image uploadée sur Cloudinary")
print("🌐 URL :", result["secure_url"])
