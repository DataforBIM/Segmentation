# =====================================================
# BASE PROMPT / BASE NEGATIVE — SDXL
# prompts/base.py
# =====================================================

# Prompt générique optimisé pour IMG2IMG / INPAINTING
# → la génération doit se limiter à la zone masquée
# → respecte la structure, la perspective et l'éclairage existants

BASE_PROMPT = (
    "raw photograph, unprocessed photo, real camera capture, "
    "authentic photography, genuine real-world scene, "
    "natural realistic textures, real surface materials, "
    "original image preservation, minimal modifications, "
    "reality-based enhancement only, "
    "photographic realism, true-to-life appearance, "
    "natural lighting conditions, real-world imperfections, "
    "preserve authenticity, maintain original character, "
    "professional photography, 8k quality"
)

# Prompt négatif universel SDXL
BASE_NEGATIVE = (
    "artifacts, visual artifacts, noise artifacts, jpeg artifacts, compression artifacts, "
    "glitches, visual glitches, rendering errors, distortion, "
    "haloing, edge artifacts, banding, posterization, "
    "unnatural textures, synthetic textures, repeated patterns, "
    "yellow tint, yellow cast, orange tint, sepia tone, color cast, wrong colors, "
    "video game graphics, game engine, gaming render, unreal engine, unity, "
    "3d render, 3d model, cgi, computer generated imagery, digital art, "
    "cartoon, anime, illustration, painting, drawing, comic, sketch, "
    "stylized, artistic style, art filter, painted effect, processed, "
    "plastic look, toy appearance, miniature effect, diorama, model, "
    "synthetic, artificial, fake, unrealistic, fantasy, "
    "perfect surfaces, too clean, overly processed, artificial sharpness, "
    "enhanced saturation, boosted colors, vibrant unrealistic colors, "
    "dramatic lighting, studio lights, artificial lighting effects, "
    "cel shading, flat shading, toon shading, posterized, "
    "low quality, low resolution, blurry, soft focus, pixelated, "
    "noise, grain, "
    "overexposed, underexposed, wrong perspective, distorted geometry, "
    "inconsistent scale, mismatched lighting, "
    "plastic, waxy, fake, CGI look, "
    "text, watermark, UI, HUD, interface elements"
)
