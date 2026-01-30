import os
import torch
from huggingface_hub import login
from diffusers import FluxPipeline

# -------------------------
# Hugging Face Login
# -------------------------
token = os.environ.get("HF_TOKEN")
if token is None:
    raise RuntimeError("HF_TOKEN non défini")

login(token=token)
print("✅ Login Hugging Face OK")

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device:", device)

# -------------------------
# Modèle FLUX
# -------------------------
MODEL_ID = "black-forest-labs/FLUX.1-dev"
# alternative rapide :
# MODEL_ID = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

pipe.to(device)

# Optimisations serveur
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

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
