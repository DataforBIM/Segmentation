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
    strength: float = 0.30,
    controlnet_scale: float = 0.80,
    guidance_scale: float = 9.0,
    num_steps: int = 40
) -> Image.Image:
    """
    Génère l'image avec SDXL + ControlNet + Refiner
    """
    
    # Construire les prompts avec le builder
    prompt, negative_prompt = build_prompts(scene_type, user_prompt)
    
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
            strength=0.20,
            guidance_scale=7.5,
            num_inference_steps=20,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        
        print("✅ Refinement terminé")
        return refined_image
    
    return base_image
