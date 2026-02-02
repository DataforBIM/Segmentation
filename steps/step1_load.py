# Load image
import requests
from io import BytesIO
from PIL import Image


def load_image(url: str) -> Image.Image:
    """Charge une image depuis une URL (Cloudinary)"""
    print(f"   📡 Téléchargement depuis: {url[:50]}...")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    print(f"   ✅ Image chargée: {image.size[0]}x{image.size[1]}")
    
    return image
