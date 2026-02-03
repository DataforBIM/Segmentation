# SDXL + ControlNet + Refiner + Inpainting - Optimisé RTX 4090 24GB
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel
)

def load_sdxl(model_id, controlnet_id, use_refiner):
    """
    Charge SDXL avec ControlNet et optionnellement le Refiner
    Optimisé pour RTX 4090 24GB
    """
    print("   🔧 Chargement de ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, 
        torch_dtype=torch.float16
    )

    print("   🎨 Chargement de SDXL Base...")
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")

    # Optimisations pour RTX 4090 24GB
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("   ⚡ XFormers activé")
    except Exception as e:
        print(f"   ⚠️  XFormers non disponible: {e}")
    
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()  # Réduit l'utilisation VRAM
    
    # Avec 24GB, on peut garder le modèle entièrement en GPU
    print("   ✅ SDXL Base chargé et optimisé")

    refiner = None
    if use_refiner:
        print("   ✨ Chargement du Refiner...")
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Optimisations pour le refiner
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except:
            pass
        refiner.enable_vae_slicing()
        refiner.enable_vae_tiling()
        print("   ✅ Refiner chargé et optimisé")

    return pipe, refiner


def load_sdxl_inpaint(use_refiner: bool = True):
    """
    Charge SDXL Inpainting pour la modification ciblée avec masque
    Optimisé pour RTX 4090 24GB
    """
    print("   🖌️  Chargement de SDXL Inpainting...")
    
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    # Optimisations
    try:
        pipe_inpaint.enable_xformers_memory_efficient_attention()
        print("   ⚡ XFormers activé (Inpainting)")
    except Exception as e:
        print(f"   ⚠️  XFormers non disponible: {e}")
    
    pipe_inpaint.enable_vae_slicing()
    pipe_inpaint.enable_vae_tiling()
    
    print("   ✅ SDXL Inpainting chargé")
    
    refiner = None
    if use_refiner:
        print("   ✨ Chargement du Refiner...")
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except:
            pass
        refiner.enable_vae_slicing()
        refiner.enable_vae_tiling()
        print("   ✅ Refiner chargé")
    
    return pipe_inpaint, refiner
