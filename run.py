# Point d'entrée - TEST SEGMENTATION ONLY
from pipeline import run_pipeline

# URL de test - remplacez par votre image
INPUT_IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770041656/sdxl_siamese_full_body_tp9mp8.png"

result = run_pipeline(
    INPUT_IMAGE_URL, 
    "Changer le fond derrière le chat",
    enable_scene_detection=True,
    enable_controlnet=True,       # ✅ ControlNet (préserve structure)
    enable_segmentation=True,     # ✅ Segmentation (masque ciblé)
    enable_sdxl=True,             # ✅ Génération
    enable_refiner=True,          # ✅ Refiner (qualité)
    enable_upscaler=True,         # ✅ Upscaling Real-ESRGAN
    segment_target="auto",
    segment_method="auto"
)

print(f"\n🖼️  Pipeline terminé!")
if result.get("mask"):
    print(f"✅ Masque généré: output/segmentation_mask.png")
    print(f"✅ Preview: output/segmentation_preview.png")
    print(f"✅ Résultat: output/output_local.png")
