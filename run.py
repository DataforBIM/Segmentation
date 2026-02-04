# Point d'entrée - TEST SEGMENTATION ONLY
from pipeline import run_pipeline

# URL de test - remplacez par votre image
INPUT_IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1769946149/1272fc67-ede0-4dbb-9d3a-f21f4ec07c79.png"

result = run_pipeline(
    INPUT_IMAGE_URL, 
    "Je veux installer un joli roof top dans la toiture",
    enable_scene_detection=True,
    enable_controlnet=True,       # ✅ ControlNet (préserve structure)
    enable_segmentation=True,     # ✅ Segmentation (masque ciblé)
    enable_sdxl=True,              # ✅ Génération SDXL activée
    enable_refiner=False,           # ✅ Refiner (qualité photoréaliste)
    enable_upscaler=False,         # ⏭️  Upscaling Real-ESRGAN
    segment_target="auto",
    segment_method="auto"
)

print(f"\n🖼️  Pipeline terminé!")
if result.get("mask"):
    print(f"✅ Masque généré: output/segmentation_mask.png")
    print(f"✅ Preview: output/segmentation_preview.png")
    print(f"✅ Résultat: output/output_local.png")
