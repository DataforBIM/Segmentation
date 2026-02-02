# Upscaling
import numpy as np
from PIL import Image


def upscale_image(image: Image.Image, upscaler, scale: int = 4) -> Image.Image:
    """
    Upscale l'image avec Real-ESRGAN
    
    Args:
        image: Image PIL à upscaler
        upscaler: Instance RealESRGANer
        scale: Facteur d'upscale (2 ou 4)
    
    Returns:
        Image upscalée
    """
    
    print(f"   🔍 Upscaling x{scale}...")
    
    # Convertir PIL → numpy
    img_np = np.array(image)
    
    # Upscale
    upscaled_np, _ = upscaler.enhance(img_np, outscale=scale)
    
    # Convertir numpy → PIL
    upscaled_image = Image.fromarray(upscaled_np)
    
    print(f"   ✅ Upscaling terminé: {upscaled_image.size[0]}x{upscaled_image.size[1]}")
    
    return upscaled_image
