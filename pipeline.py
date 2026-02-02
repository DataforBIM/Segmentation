# Orchestration centrale
from PIL import Image
from config.settings import *
from models.blip import detect_scene_type
from steps.step1_load import load_image
from steps.step2_preprocess import make_canny, compute_output_size
from steps.step4_upscale import upscale_image
from steps.step5_upload import upload_to_cloudinary


def run_pipeline(
    image_url: str, 
    user_prompt: str,
    # Contrôle des étapes du pipeline
    enable_scene_detection: bool = True,
    enable_controlnet: bool = True,
    enable_sdxl: bool = False,
    enable_refiner: bool = False,
    enable_upscaler: bool = False,
    enable_upload: bool = False
) -> dict:
    """
    Pipeline complet de génération d'images architecturales
    
    Args:
        image_url: URL de l'image d'entrée (Cloudinary)
        user_prompt: Prompt utilisateur
        enable_scene_detection: Activer la détection de scène BLIP
        enable_controlnet: Activer ControlNet (Canny)
        enable_sdxl: Activer la génération SDXL
        enable_refiner: Activer le refiner SDXL
        enable_upscaler: Activer l'upscaling Real-ESRGAN
        enable_upload: Activer l'upload vers Cloudinary
    
    Returns:
        Dict avec l'image finale et les métadonnées
    """
    
    print("="*60)
    print("🚀 DÉMARRAGE DU PIPELINE")
    print("="*60)
    
    # Étape 1: Chargement
    print("\n📥 Étape 1: Chargement de l'image")
    current_image = load_image(image_url)
    last_step = "load"
    
    # Étape 2: Détection de scène
    scene_type = "EXTERIOR"  # Valeur par défaut
    if enable_scene_detection:
        print("\n🧠 Étape 2: Détection de scène")
        scene_type = detect_scene_type(current_image)
        print(f"   🎯 Scène détectée: {scene_type}")
    else:
        print("\n⏭️  Étape 2: Détection de scène désactivée (utilisation: EXTERIOR)")
    
    # Étape 3: Prétraitement
    print("\n🎨 Étape 3: Prétraitement")
    width, height = compute_output_size(current_image, MAX_SIZE)
    print(f"   📐 Dimensions: {width}x{height}")
    
    control_image = None
    if enable_controlnet:
        control_image = make_canny(current_image)
        print("   ✅ ControlNet (Canny) activé")
    else:
        print("   ⏭️  ControlNet désactivé")
    
    # Étape 4 & 5: Génération SDXL
    if enable_sdxl:
        print("\n🔧 Étape 4: Chargement des modèles SDXL")
        from models.sdxl import load_sdxl
        from steps.step3_generate import generate_with_sdxl
        
        pipe, refiner = load_sdxl(
            SDXL_MODEL, 
            CONTROLNET_MODEL, 
            enable_refiner and USE_REFINER
        )
        
        print("\n🎭 Étape 5: Génération SDXL")
        current_image = generate_with_sdxl(
            image=current_image,
            control_image=control_image,
            pipe=pipe,
            refiner=refiner if enable_refiner else None,
            scene_type=scene_type,
            user_prompt=user_prompt,
            width=width,
            height=height,
            seed=SEED
        )
        
        # Mettre à jour la dernière étape
        if enable_refiner:
            last_step = "refiner"
        else:
            last_step = "sdxl"
    else:
        print("\n⏭️  Étapes 4-5: SDXL désactivé")
    
    # Étape 6: Upscaling
    if enable_upscaler and USE_UPSCALER:
        print("\n🔍 Étape 6: Upscaling Real-ESRGAN")
        from models.upscaler import load_upscaler
        
        upscaler = load_upscaler()
        current_image = upscale_image(current_image, upscaler)
        last_step = "upscaler"
    else:
        print("\n⏭️  Étape 6: Upscaling désactivé")
    
    # L'image finale est toujours le résultat de la dernière étape activée
    final_image = current_image
    
    # Étape 7: Upload Cloudinary
    cloudinary_url = None
    if enable_upload:
        print("\n☁️  Étape 7: Upload vers Cloudinary")
        cloudinary_url = upload_to_cloudinary(
            final_image,
            folder="sdxl_outputs/pipeline"
        )
    else:
        print("\n⏭️  Étape 7: Upload Cloudinary désactivé")
        # Sauvegarder localement à la place
        local_path = "output_local.png"
        final_image.save(local_path)
        print(f"   💾 Image sauvegardée localement: {local_path}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE TERMINÉ")
    print("="*60)
    print(f"📸 Image finale générée par: {last_step}")
    if cloudinary_url:
        print(f"🌐 URL finale: {cloudinary_url}")
    else:
        print(f"💾 Fichier local: output_local.png")
    
    return {
        "image": final_image,
        "scene_type": scene_type,
        "cloudinary_url": cloudinary_url,
        "dimensions": final_image.size,
        "last_step_executed": last_step,
        "steps_executed": {
            "scene_detection": enable_scene_detection,
            "controlnet": enable_controlnet,
            "sdxl": enable_sdxl,
            "refiner": enable_refiner,
            "upscaler": enable_upscaler,
            "upload": enable_upload
        }
    }
