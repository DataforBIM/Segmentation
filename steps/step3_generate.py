# SDXL generation
import torch
from PIL import Image
from prompts.builders import build_prompts


def generate_with_sdxl(
    image: Image.Image,
    control_image: Image.Image,
    pipe,
    refiner,
    prompt_config: dict,  # NOUVEAU: Configuration modulaire du prompt
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
    
    # Construire les prompts avec le builder modulaire
    prompt, negative_prompt = build_prompts(**prompt_config)
    
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


def generate_aerial_multipass(
    image: Image.Image,
    control_images: dict,
    pipe,
    refiner,
    user_prompt: str,
    width: int,
    height: int,
    seed: int,
    aerial_elements: list,
    prompt_config: dict  # NOUVEAU: Configuration modulaire du prompt
) -> Image.Image:
    """
    🚁 Génération SDXL en 3 passes pour scènes aériennes
    
    Passe 1 - STRUCTURE: walls + roof (denoise=0.50, depth=ON)
    Passe 2 - OUVERTURES: windows + doors (denoise=0.20, depth=OFF)
    Passe 3 - DÉTAILS: ornementation + road + sidewalk (denoise=0.28)
    
    Args:
        image: Image d'entrée
        control_images: Dict avec depth, canny, etc.
        pipe: Pipeline SDXL
        refiner: Refiner SDXL
        user_prompt: Prompt utilisateur
        width, height: Dimensions
        seed: Seed aléatoire
        aerial_elements: Liste des éléments détectés ["walls", "roof", "window", ...]
        prompt_config: Configuration modulaire du prompt
    
    Returns:
        Image finale après 3 passes
    """
    print("\n🚁 === GÉNÉRATION AÉRIENNE MULTI-PASS (3 passes) ===")
    
    current = image.copy()
    
    # === PASSE 1: STRUCTURE (walls + roof) ===
    print("\n📐 PASSE 1/3: STRUCTURE (walls + roof)")
    print("   Paramètres: denoise=0.50, depth=ON, controlnet=1.2")
    
    structure_elements = ["walls", "roof"]
    # Pour vue aérienne: toujours exécuter même si non détecté
    
    current = generate_with_sdxl(
        image=current,
        control_image=control_images.get("depth"),
        pipe=pipe,
        refiner=None,  # Pas de refiner entre les passes
        prompt_config=prompt_config,  # Utiliser la configuration modulaire
        width=width,
        height=height,
        seed=seed,
        strength=0.50,  # Denoise élevé pour structure
        controlnet_scale=1.2,
        guidance_scale=5.0,
        num_steps=40,
        aerial_elements=structure_elements  # Passer tous les éléments par défaut
    )
    print(f"   ✅ Structure générée (mask par défaut)")
    
    # === PASSE 2: OUVERTURES (windows + doors) ===
    print("\n🚪 PASSE 2/3: OUVERTURES (windows + doors)")
    print("   Paramètres: denoise=0.20, depth=OFF, controlnet=1.2")
    
    opening_elements = ["window", "door"]
    # Pour vue aérienne: toujours exécuter même si non détecté
    
    current = generate_with_sdxl(
        image=current,
        control_image=None,  # Depth OFF pour ouvertures
        pipe=pipe,
        refiner=None,
        prompt_config=prompt_config,  # Utiliser la configuration modulaire
        width=width,
        height=height,
        seed=seed,
        strength=0.20,  # Denoise faible pour préserver
        controlnet_scale=1.2,
        guidance_scale=5.0,
        num_steps=40,
        aerial_elements=opening_elements  # Passer tous les éléments par défaut
    )
    print(f"   ✅ Ouvertures générées (mask par défaut)")
    
    # === PASSE 3: DÉTAILS/CONTEXTE (ornementation + road + sidewalk) ===
    print("\n✨ PASSE 3/3: DÉTAILS/CONTEXTE (ornementation + road + sidewalk)")
    print("   Paramètres: denoise=0.28, depth=ON, controlnet=1.2")
    
    detail_elements = ["ornementation", "road", "sidewalk", "road_markings", "car", "vegetation", "parking"]
    # Pour vue aérienne: toujours exécuter même si non détecté
    
    current = generate_with_sdxl(
        image=current,
        control_image=control_images.get("depth"),  # Depth ON pour contexte
        pipe=pipe,
        refiner=refiner,  # Refiner sur la dernière passe uniquement
        prompt_config=prompt_config,  # Utiliser la configuration modulaire
        width=width,
        height=height,
        seed=seed,
        strength=0.28,  # Denoise modéré
        controlnet_scale=1.2,
        guidance_scale=5.0,
        num_steps=40,
        aerial_elements=detail_elements  # Passer tous les éléments par défaut
    )
    print(f"   ✅ Détails générés (mask par défaut)")
    
    print("\n✅ === 3 PASSES TERMINÉES ===")
    return current
