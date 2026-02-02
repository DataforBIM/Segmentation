# Cloudinary
import os
from datetime import datetime
from PIL import Image
import cloudinary.uploader


def upload_to_cloudinary(
    image: Image.Image,
    folder: str = "sdxl_outputs",
    public_id_prefix: str = "archviz"
) -> str:
    """
    Upload l'image vers Cloudinary
    
    Args:
        image: Image PIL à uploader
        folder: Dossier Cloudinary
        public_id_prefix: Préfixe pour le public_id
    
    Returns:
        URL sécurisée de l'image uploadée
    """
    
    # Sauvegarder temporairement
    temp_path = "temp_output.png"
    image.save(temp_path)
    
    # Upload
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    public_id = f"{public_id_prefix}_{timestamp}"
    
    print(f"   📤 Upload vers: {folder}/{public_id}")
    
    result = cloudinary.uploader.upload(
        temp_path,
        folder=folder,
        public_id=public_id,
        overwrite=True
    )
    
    # Nettoyer
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    url = result["secure_url"]
    print(f"   ✅ Upload terminé")
    
    return url
