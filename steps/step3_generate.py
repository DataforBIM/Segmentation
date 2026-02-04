# SDXL generation
import torch
from PIL import Image
from prompts.builders import build_prompts


def generate_with_sdxl(
    image: Image.Image,
    control_image: Image.Image,
    pipe,
    refiner,
    scene_type: str,
    user_prompt: str,
    width: int,
    height: int,
    seed: int = 123456,
    strength: float = 0.20,  # Encore plus faible pour éviter les artefacts
    controlnet_scale: float = 1.2,  # Augmenté pour depth plus fort
    guidance_scale: float = 5.0,  # Très réduit pour éviter les artefacts
    num_steps: int = 40,  # Réduit pour moins de transformation
    aerial_elements: list[str] = None  # NOUVEAU: éléments aériens
) -> Image.Image:
    """
    Génère l'image avec SDXL + ControlNet + Refiner
    Paramètres optimisés pour minimiser les artefacts
    """
    
    # Construire les prompts avec le builder (avec éléments aériens si disponibles)
    prompt, negative_prompt = build_prompts(scene_type, user_prompt, aerial_elements=aerial_elements)
    
    print(f"\n🎨 Prompt final: {prompt[:100]}...")
    print(f"🚫 Negative: {negative_prompt[:100]}...")
    
    # Génération avec ControlNet
    generator = torch.Generator("cuda").manual_seed(seed)
    
    base_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_image=control_image,
        strength=strength,
        controlnet_conditioning_scale=controlnet_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    print("✅ Génération SDXL terminée")
    
    # Refinement si disponible
    if refiner:
        print("🔧 Application du refiner...")
        
        refined_image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            strength=0.15,  # Minimum safe value for VAE
            guidance_scale=5.0,
            num_inference_steps=15,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        
        print("✅ Refinement terminé")
        return refined_image
    
    return base_image
