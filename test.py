import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"
# MODEL_ID = "black-forest-labs/FLUX.1-schnell"  # plus rapide

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="balanced"   # ← suffit à lui seul
)

# Optimisation mémoire OK
pipe.enable_attention_slicing()

print("✅ FLUX chargé")

prompt = (
    "minimalist apartment floor plan, pure white background, "
    "architectural drawing, top view, clean lines, no shadows"
)

image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_inference_steps=30,
    height=1024,
    width=1024,
).images[0]

image.save("flux_output.png")
print("✅ Image générée : flux_output.png")
