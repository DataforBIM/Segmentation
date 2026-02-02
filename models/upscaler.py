# Real-ESRGAN
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def load_upscaler(model_path: str = None, scale: int = 4):
    """
    Charge le modèle Real-ESRGAN pour l'upscaling
    
    Args:
        model_path: Chemin vers le modèle (optionnel, télécharge automatiquement)
        scale: Facteur d'upscale (2 ou 4)
    
    Returns:
        Instance RealESRGANer
    """
    
    print("   📦 Chargement de Real-ESRGAN...")
    
    # Architecture du modèle RRDBNet
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )
    
    # Créer l'upscaler
    upscaler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        gpu_id=0 if torch.cuda.is_available() else None
    )
    
    print("   ✅ Real-ESRGAN chargé")
    
    return upscaler
