import torch
from diffusers import FluxPipeline

# -------------------------
# Modèle FLUX
# -------------------------
MODEL_ID = "black-forest-labs/FLUX.1-dev"
# alternative rapide :
# MODEL_ID = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,     # plus stable que BF16 en test
    device_map="auto",             # gestion automatique CPU/GPU
)

# Offload CPU (OBLIGATOIRE pour FLUX)
pipe.enable_model_cpu_offload()

# Optimisations mémoire
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
# Sauvegarde
# -------------------------
output_path = "flux_output.png"
image.save(output_path)

print(f"✅ Image générée : {output_path}")
