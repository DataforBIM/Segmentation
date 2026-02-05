# SDXL Inpainting generation
import torch
from PIL import Image
from prompts.builders import build_prompts


def generate_with_inpainting(
    image: Image.Image,
    mask: Image.Image,
    pipe_inpaint,
    refiner,
    prompt_config: dict,  # NOUVEAU: Configuration modulaire du prompt
    width: int,
    height: int,
    seed: int = 123456,
    strength: float = 0.50,  # Très faible pour éviter les artefacts
    guidance_scale: float = 5.0,  # Très réduit pour éviter les artefacts
    num_steps: int = 50,
    aerial_elements: list[str] = None  # NOUVEAU: éléments détectés pour scènes aériennes
) -> Image.Image:
    """
    Génère l'image avec SDXL Inpainting
    Modifie UNIQUEMENT la zone masquée
    
    Args:
        image: Image originale
        mask: Masque (blanc = zone à modifier)
        pipe_inpaint: Pipeline SDXL Inpainting
        refiner: Pipeline Refiner (optionnel)
        prompt_config: Configuration modulaire du prompt
        width, height: Dimensions de sortie
        seed: Seed pour la reproductibilité
        strength: Force de modification (0.99 = remplacement quasi-total)
        guidance_scale: Adhérence au prompt
        num_steps: Nombre d'étapes d'inférence
        aerial_elements: Liste des éléments détectés pour scènes aériennes
    
    Returns:
        Image avec la zone masquée modifiée
    """
    
    # Construire les prompts avec le builder modulaire
    prompt, negative_prompt = build_prompts(**prompt_config)
    
    print(f"\n🎨 Prompt final: {prompt[:100]}...")
    print(f"🚫 Negative: {negative_prompt[:100]}...")
    
    # Redimensionner image et masque à la taille cible
    image_resized = image.resize((width, height), Image.Resampling.LANCZOS)
    mask_resized = mask.resize((width, height), Image.Resampling.NEAREST)
    
    # Convertir le masque en RGB si nécessaire
    if mask_resized.mode != "RGB":
        mask_rgb = Image.new("RGB", mask_resized.size)
        mask_rgb.paste(mask_resized)
        mask_resized = mask_rgb
    
    # Génération avec Inpainting
    generator = torch.Generator("cuda").manual_seed(seed)
    
    print(f"   🖌️  Inpainting avec strength={strength}...")
    
    base_image = pipe_inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_resized,
        mask_image=mask_resized,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    
    print("✅ Inpainting SDXL terminé")
    
    # Refinement si disponible
    if refiner:
        print("🔧 Application du refiner...")
        
        refined_image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            strength=0.05,  # Extrêmement léger pour éviter les artefacts
            guidance_scale=4.5,  # Très réduit
            num_inference_steps=10,  # Très réduit
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        
        print("✅ Refinement terminé")
        return refined_image
    
    return base_image


def generate_with_controlnet_inpaint(
    image: Image.Image,
    mask: Image.Image,
    control_image: Image.Image,
    pipe,
    refiner,
    prompt_config: dict,  # NOUVEAU: Configuration modulaire du prompt
    width: int,
    height: int,
    seed: int = 123456,
    strength: float = 0.85,
    controlnet_scale: float = 0.6,
    guidance_scale: float = 10.0,
    num_steps: int = 50,
    aerial_elements: list[str] = None  # NOUVEAU: éléments aériens
) -> Image.Image:
    """
    Génère avec ControlNet + Masque de fusion manuel
    Combine les avantages de ControlNet et de l'inpainting
    
    Cette méthode:
    1. Génère une nouvelle image avec ControlNet
    2. Fusionne avec l'originale en utilisant le masque
    """
    from steps.step3_generate import generate_with_sdxl
    
    print("   🎨 Génération ControlNet + Fusion masquée...")
    
    # Générer l'image complète avec ControlNet
    generated = generate_with_sdxl(
        image=image,
        control_image=control_image,
        pipe=pipe,
        refiner=refiner,
        prompt_config=prompt_config,  # Utiliser la configuration modulaire
        width=width,
        height=height,
        seed=seed,
        strength=strength,
        controlnet_scale=controlnet_scale,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        aerial_elements=aerial_elements  # Passer les éléments aériens
    )
    
    # Fusionner avec le masque
    print("   🔀 Fusion avec le masque...")
    
    # Redimensionner à la même taille
    original_resized = image.resize(generated.size, Image.Resampling.LANCZOS)
    mask_resized = mask.resize(generated.size, Image.Resampling.LANCZOS).convert("L")
    
    # Composite: original où masque=0, généré où masque=255
    result = Image.composite(generated, original_resized, mask_resized)
    
    print("   ✅ Fusion terminée")
    
    return result
