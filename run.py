# Point d'entrée
from pipeline import run_pipeline

# URL de test - remplacez par votre image
INPUT_IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770041656/sdxl_siamese_full_body_tp9mp8.png"

result = run_pipeline(
    INPUT_IMAGE_URL, 
    "Améliorer la qualité de l'image",
    enable_controlnet=True  # Activer ControlNet
)

result["image"].save("output.png")
