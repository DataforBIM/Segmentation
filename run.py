# Point d'entrée
from pipeline import run_pipeline

# URL de test - remplacez par votre image
INPUT_IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770041510/969ee8_76ce65d86468468d85240537df898890_mv2_ywh2ym.avif"

result = run_pipeline(INPUT_IMAGE_URL, "Améliorer la qualité de l'image")

result["image"].save("output.png")
