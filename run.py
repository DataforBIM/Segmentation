# Point d'entrée
from steps.load import load_image_from_url
from pipeline import run_pipeline

image = load_image_from_url(INPUT_IMAGE_URL)
result = run_pipeline(image, "Améliorer la qualité de l'image")

result.save("output.png")
