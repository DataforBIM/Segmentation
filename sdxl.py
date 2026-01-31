import os
import torch
import cloudinary
import cloudinary.uploader
from diffusers import StableDiffusionXLPipeline

# =====================================================
# Configuration Cloudinary (via variables d’environnement)
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

# Optimisations GPU (A100 / grosses cartes)
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

print("✅ SDXL RealVis XL chargé avec succès")

# =====================================================
# Prompt – VUE ARCHITECTURALE RÉALISTE (EXTÉRIEUR)
# =====================================================
prompt = (
    "Photographie architecturale réaliste d’un bâtiment contemporain, "
    "vue extérieure soigneusement cadrée, "

    "architecture moderne haut de gamme, lignes épurées, "
    "volumes lisibles et bien proportionnés, "
    "façade en béton brut, verre clair et métal, "
    "détails constructifs précis, joints visibles, "

    "vue en perspective à hauteur d’homme, "
    "camera eye level, focal length 24mm, "
    "wide shot, building fully visible, no crop, "
    "composition architecturale équilibrée, "

    "éclairage naturel réaliste, lumière douce de fin de journée, "
    "ombres cohérentes, global illumination naturelle, "

    "environnement urbain sobre, sol minéral, "
    "végétation intégrée réaliste, arbres bien proportionnés, "

    "style photographie d’architecture professionnelle, "
    "ultra realistic, high detail, sharp focus, "
    "physically accurate lighting, real materials"
)

# =====================================================
# Negative Prompt – éviter les rendus IA irréalistes
# =====================================================
negative_prompt = (
    "cartoon, illustration, anime, painting, "
    "3d render, cgi, unreal engine look, "

    "distorted perspective, warped lines, "
    "broken geometry, impossible architecture, "
    "floating buildings, unrealistic scale, "

    "close-up, cropped building, partial view, "
    "fish-eye, extreme wide angle distortion, "

    "overexposed, underexposed, flat lighting, "
    "blurry, low detail, noise, "

    "people in foreground, cars too close, "
    "text, logo, watermark"
)

# =====================================================
# Génération de l’image
# =====================================================
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=6.0,        # équilibre fidélité / liberté
    num_inference_steps=30,    # qualité stable pour l’architecture
    width=1024,
    height=1024
).images[0]

# =====================================================
# Sauvegarde locale
# =====================================================
local_path = "sdxl_architectural_view.png"
image.save(local_path)

# =====================================================
# Upload Cloudinary
# =====================================================
result = cloudinary.uploader.upload(
    local_path,
    folder="sdxl_outputs",
    public_id="sdxl_architectural_view",
    overwrite=True
)

print("✅ Image générée et uploadée sur Cloudinary")
print("🌐 URL :", result["secure_url"])
`
