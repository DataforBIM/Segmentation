# Point d'entrée
from pipeline import run_pipeline

# URL de test - remplacez par votre image
INPUT_IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1769938551/BAC_CHAMBRE_wd3mo8.jpg"

result = run_pipeline(
    INPUT_IMAGE_URL, 
    "Améliorer la qualité de l'image",
    enable_controlnet=True,  # Activer ControlNet
    enable_sdxl=True,        # Activer SDXL
    enable_refiner=True      # Activer Refiner
)

result["image"].save("output.png")
