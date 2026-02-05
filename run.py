# Point d'entrée - TEST SEGMENTATION ONLY
from pipeline import run_pipeline

# URL de test - remplacez par votre image
INPUT_IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

result = run_pipeline(
    INPUT_IMAGE_URL, 
    "Change la couleur de la façade",
    # Configuration du prompt modulaire (auto-détecté si None)
    scene_structure=None,         # "exterior" sera auto-détecté
    subject=None,                 # "facade" sera auto-détecté
    auto_detect_prompt=True,      # ✅ Auto-détection depuis le prompt
    # Contrôle des étapes
    enable_controlnet=True,       # ✅ ControlNet (préserve structure)
    enable_segmentation=True,     # ✅ Segmentation (masque ciblé)
    enable_sdxl=True,            # ⏭️  Génération SDXL
    enable_refiner=False,         # ⏭️  Refiner (qualité photoréaliste)
    enable_upscaler=False,        # ⏭️  Upscaling Real-ESRGAN
    segment_target="auto",
    segment_method="auto"
)

print(f"\n🖼️  Pipeline terminé!")
if result.get("mask"):
    print(f"✅ Masque généré: output/segmentation_mask.png")
    print(f"✅ Preview: output/segmentation_preview.png")
    print(f"✅ Résultat: output/output_local.png")
