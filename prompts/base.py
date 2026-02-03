# =====================================================
# BASE PROMPT / BASE NEGATIVE — SDXL
# prompts/base.py
# =====================================================

# Prompt générique optimisé pour IMG2IMG / INPAINTING
# → la génération doit se limiter à la zone masquée
# → respecte la structure, la perspective et l'éclairage existants

BASE_PROMPT = (
    "photorealistic, ultra-detailed, high quality, "
    "real-world materials and textures, "
    "physically based rendering, realistic surface response, "
    "natural lighting, accurate shadows, "
    "correct perspective, consistent scale, "
    "clean geometry, sharp focus, "
    "seamless integration with surrounding area, "
    "matches existing environment, "
    "professional photography, 8k quality"
)

# Prompt négatif universel SDXL
BASE_NEGATIVE = (
    "cartoon, illustration, anime, painting, drawing, "
    "stylized, fantasy, surreal, "
    "low quality, low resolution, blurry, soft focus, "
    "noise, grain, jpeg artifacts, compression artifacts, "
    "overexposed, underexposed, flat lighting, harsh shadows, "
    "wrong perspective, distorted geometry, warped, "
    "inconsistent scale, mismatched lighting, "
    "plastic, waxy, fake, CGI look"
)
