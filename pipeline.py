# Orchestration centrale
from PIL import Image
from config.settings import *
from models.blip import detect_scene_type
from steps.step1_load import load_image
from steps.step2_preprocess import compute_output_size
from steps.step4_upscale import upscale_image
from steps.step5_upload import upload_to_cloudinary


def run_pipeline(
    image_url: str, 
    user_prompt: str,
    # Contrôle des étapes du pipeline
    enable_scene_detection: bool = True,
    enable_controlnet: bool = True,
    enable_segmentation: bool = True,  # NOUVELLE: Segmentation SAM2
    enable_sdxl: bool = False,
    enable_refiner: bool = False,
    enable_upscaler: bool = False,
    enable_upload: bool = False,
    # Options de segmentation
    segment_target: str = "auto",  # "auto" = détection automatique depuis le prompt
    segment_method: str = "auto"
) -> dict:
    """
    Pipeline complet de génération d'images architecturales
    
    Args:
        image_url: URL de l'image d'entrée (Cloudinary)
        user_prompt: Prompt utilisateur
        enable_scene_detection: Activer la détection de scène BLIP
        enable_controlnet: Activer ControlNet (Canny/Depth)
        enable_segmentation: Activer la segmentation SAM2/SegFormer
        enable_sdxl: Activer la génération SDXL
        enable_refiner: Activer le refiner SDXL
        enable_upscaler: Activer l'upscaling Real-ESRGAN
        enable_upload: Activer l'upload vers Cloudinary
        segment_target: Cible de segmentation ("floor", "wall", etc.)
        segment_method: Méthode ("auto", "points", "box")
    
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
    
    control_images = {}
    if enable_controlnet:
        from steps.step2_preprocess import make_canny, make_depth, make_openpose, make_normal
        
        # Générer tous les pass de ControlNet
        print("   🎨 Génération de tous les pass ControlNet...")
        
        # Pass 1: Canny (contours) - Très soft pour scènes aériennes
        try:
            if scene_type == "AERIAL":
                # Canny très soft pour préserver les détails fins en aérien
                control_images["canny"] = make_canny(current_image, save_path="output/controlnet_canny.png", low_threshold=30, high_threshold=80)
            else:
                # Canny normal pour autres scènes
                control_images["canny"] = make_canny(current_image, save_path="output/controlnet_canny.png")
        except Exception as e:
            print(f"   ⚠️  Erreur Canny: {e}")
        
        # Pass 2: Depth (profondeur)
        try:
            control_images["depth"] = make_depth(current_image, save_path="output/controlnet_depth.png")
        except Exception as e:
            print(f"   ⚠️  Erreur Depth: {e}")
        
        # Pass 3: OpenPose (poses) - Skip for AERIAL scenes
        if scene_type != "AERIAL":
            try:
                control_images["openpose"] = make_openpose(current_image, save_path="output/controlnet_openpose.png")
            except Exception as e:
                print(f"   ⚠️  Erreur OpenPose: {e}")
        else:
            print(f"   ⏭️  OpenPose désactivé pour scènes aériennes")
        
        # Pass 4: Normal (normales) - Skip for AERIAL scenes
        if scene_type != "AERIAL":
            try:
                control_images["normal"] = make_normal(current_image, save_path="output/controlnet_normal.png")
            except Exception as e:
                print(f"   ⚠️  Erreur Normal: {e}")
        else:
            print(f"   ⏭️  Normal map désactivée pour scènes aériennes")
        
        print(f"   ✅ {len(control_images)} pass ControlNet générés")
    else:
        print("   ⏭️  ControlNet désactivé")
    
    # Étape 4: Segmentation SAM2/SegFormer (NOUVELLE)
    mask = None
    aerial_elements = None  # Pour stocker les éléments détectés en mode aérien
    
    if enable_segmentation and USE_SEGMENTATION:
        print("\n🧠 Étape 4: Segmentation SAM2/SegFormer")
        from steps.step2b_segment import segment_target_region, create_masked_image, load_aerial_metadata
        from prompts.target_detection import detect_segment_target, get_target_description
        
        # Détection automatique de la cible si nécessaire
        if segment_target == "auto":
            # Passer la scène détectée pour un meilleur filtrage
            segment_target = detect_segment_target(user_prompt, scene_type=scene_type)
            print(f"   🎯 Cible détectée automatiquement: {get_target_description(segment_target)}")
        else:
            print(f"   🎯 Cible spécifiée manuellement: {get_target_description(segment_target)}")
        
        mask = segment_target_region(
            image=current_image,
            target=segment_target,
            method=segment_method,
            scene_type=scene_type,  # Passer la scène détectée
            dilate=SEGMENT_DILATE,
            feather=SEGMENT_FEATHER,
            save_path="output/segmentation_mask.png"
        )
        
        # Pour les scènes aériennes, charger les métadonnées des éléments détectés
        if scene_type == "AERIAL":
            aerial_elements = load_aerial_metadata("output/segmentation_mask.png")
            if aerial_elements:
                print(f"   ✅ Éléments aériens chargés: {len(aerial_elements)} types")
        
        # Sauvegarder une preview du masque sur l'image
        create_masked_image(
            current_image, 
            mask, 
            save_path="output/segmentation_preview.png"
        )
    else:
        print("\n⏭️  Étape 4: Segmentation désactivée")
    
    # Étape 5 & 6: Génération SDXL (Inpainting ou ControlNet)
    if enable_sdxl:
        print("\n🔧 Étape 5: Chargement des modèles SDXL")
        
        # Choix du mode: Inpainting (avec masque) ou ControlNet (global)
        if mask is not None and USE_INPAINTING:
            # Mode INPAINTING - Modification ciblée
            print("   🎯 Mode: INPAINTING (modification ciblée)")
            from models.sdxl import load_sdxl_inpaint
            from steps.step3b_inpaint import generate_with_inpainting
            
            pipe_inpaint, refiner = load_sdxl_inpaint(enable_refiner and USE_REFINER)
            
            print("\n🎭 Étape 6: Génération SDXL Inpainting")
            
            current_image = generate_with_inpainting(
                image=current_image,
                mask=mask,
                pipe_inpaint=pipe_inpaint,
                refiner=refiner if enable_refiner else None,
                scene_type=scene_type,
                user_prompt=user_prompt,
                width=width,
                height=height,
                seed=SEED,
                aerial_elements=aerial_elements  # Passer les éléments aériens
            )
            
        elif mask is not None:
            # Mode CONTROLNET + FUSION avec masque
            print("   🎯 Mode: CONTROLNET + Fusion masquée")
            from models.sdxl import load_sdxl
            from steps.step3b_inpaint import generate_with_controlnet_inpaint
            
            pipe, refiner = load_sdxl(
                SDXL_MODEL, 
                CONTROLNET_MODEL, 
                enable_refiner and USE_REFINER
            )
            
            print("\n🎭 Étape 6: Génération ControlNet + Fusion")
            control_img = control_images.get("depth") if control_images else None
            
            current_image = generate_with_controlnet_inpaint(
                image=current_image,
                mask=mask,
                control_image=control_img,
                pipe=pipe,
                refiner=refiner if enable_refiner else None,
                scene_type=scene_type,
                user_prompt=user_prompt,
                width=width,
                height=height,
                seed=SEED,
                aerial_elements=aerial_elements  # Passer les éléments aériens
            )
            
        else:
            # Mode CONTROLNET classique (sans masque)
            print("   🎯 Mode: CONTROLNET classique")
            from models.sdxl import load_sdxl
            from steps.step3_generate import generate_with_sdxl
            
            pipe, refiner = load_sdxl(
                SDXL_MODEL, 
                CONTROLNET_MODEL, 
                enable_refiner and USE_REFINER
            )
            
            print("\n🎭 Étape 6: Génération SDXL")
            control_img = control_images.get("depth") if control_images else None
            print(f"   🎛️  ControlNet utilisé: {'Depth' if control_img else 'None'}")
            
            current_image = generate_with_sdxl(
                image=current_image,
                control_image=control_img,
                pipe=pipe,
                refiner=refiner if enable_refiner else None,
                scene_type=scene_type,
                user_prompt=user_prompt,
                width=width,
                height=height,
                seed=SEED,
                aerial_elements=aerial_elements  # Passer les éléments aériens
            )
        
        # Mettre à jour la dernière étape
        if enable_refiner:
            last_step = "refiner"
        else:
            last_step = "sdxl"
    else:
        print("\n⏭️  Étapes 5-6: SDXL désactivé")
    
    # Étape 7: Upscaling
    if enable_upscaler and USE_UPSCALER:
        print("\n🔍 Étape 7: Upscaling Real-ESRGAN")
        from models.upscaler import load_upscaler
        
        upscaler = load_upscaler()
        current_image = upscale_image(current_image, upscaler)
        last_step = "upscaler"
    else:
        print("\n⏭️  Étape 7: Upscaling désactivé")
    
    # L'image finale est toujours le résultat de la dernière étape activée
    final_image = current_image
    
    # Étape 8: Upload Cloudinary
    cloudinary_url = None
    if enable_upload:
        print("\n☁️  Étape 8: Upload vers Cloudinary")
        cloudinary_url = upload_to_cloudinary(
            final_image,
            folder="sdxl_outputs/pipeline"
        )
    else:
        print("\n⏭️  Étape 8: Upload Cloudinary désactivé")
        # Sauvegarder localement à la place
        local_path = "output/output_local.png"
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
        "mask": mask,
        "cloudinary_url": cloudinary_url,
        "dimensions": final_image.size,
        "last_step_executed": last_step,
        "steps_executed": {
            "scene_detection": enable_scene_detection,
            "controlnet": enable_controlnet,
            "segmentation": enable_segmentation,
            "sdxl": enable_sdxl,
            "refiner": enable_refiner,
            "upscaler": enable_upscaler,
            "upload": enable_upload
        }
    }
