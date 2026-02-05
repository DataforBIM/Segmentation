# Orchestration centrale
from PIL import Image
from config.settings import *
from steps.step1_load import load_image
from steps.step2_preprocess import compute_output_size
from steps.step4_upscale import upscale_image
from steps.step5_upload import upload_to_cloudinary


def run_pipeline(
    image_url: str, 
    user_prompt: str,
    # NOUVEAU: Configuration du prompt modulaire
    scene_structure: str = None,  # interior, exterior, aerial, landscape, detail (auto si None)
    subject: str = None,  # building, facade, interior_space, etc. (auto si None)
    environment: str = None,  # urban, residential, park, etc. (auto si None)
    camera: list[str] | str = None,  # eye_level, wide_angle, etc. (auto si None)
    lighting: str = None,  # natural_daylight, golden_hour, etc. (auto si None)
    materials: list[str] | str = None,  # concrete, glass, wood, etc. (auto si None)
    style: list[str] | str = None,  # photorealistic, architectural_photo, etc. (auto si None)
    auto_detect_prompt: bool = True,  # Auto-détection des paramètres depuis le prompt
    # Contrôle des étapes du pipeline
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
    Pipeline complet de génération d'images architecturales avec prompts modulaires
    
    Args:
        image_url: URL de l'image d'entrée (Cloudinary)
        user_prompt: Prompt utilisateur
        scene_structure: Structure de scène (auto-détecté si None)
        subject: Sujet principal (auto-détecté si None)
        environment: Environnement (auto-détecté si None)
        camera: Paramètres caméra (auto-détecté si None)
        lighting: Conditions d'éclairage (auto-détecté si None)
        materials: Matériaux (auto-détecté si None)
        style: Style photographique (auto-détecté si None)
        auto_detect_prompt: Active l'auto-détection des paramètres
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
    print("🚀 DÉMARRAGE DU PIPELINE - PROMPT MODULAIRE")
    print("="*60)
    
    # Étape 1: Chargement
    print("\n📥 Étape 1: Chargement de l'image")
    current_image = load_image(image_url)
    last_step = "load"
    
    # Étape 2: Configuration du prompt modulaire
    print("\n🧠 Étape 2: Configuration du prompt modulaire")
    print(f"   📝 Prompt utilisateur: {user_prompt}")
    
    # Stocker la configuration du prompt pour l'utiliser plus tard
    prompt_config = {
        "user_prompt": user_prompt,
        "scene_structure": scene_structure,
        "subject": subject,
        "environment": environment,
        "camera": camera,
        "lighting": lighting,
        "materials": materials,
        "style": style,
        "auto_detect": auto_detect_prompt
    }
    
    if auto_detect_prompt and not scene_structure:
        print(f"   🎯 Mode: Auto-détection des paramètres depuis le prompt")
    else:
        print(f"   🎯 Mode: Configuration manuelle")
        if scene_structure:
            print(f"      - Structure: {scene_structure}")
        if subject:
            print(f"      - Sujet: {subject}")
        if environment:
            print(f"      - Environnement: {environment}")
    
    # Étape 3: Prétraitement
    print("\n🎨 Étape 3: Prétraitement")
    width, height = compute_output_size(current_image, MAX_SIZE)
    print(f"   📐 Dimensions: {width}x{height}")
    
    control_images = {}
    if enable_controlnet:
        from steps.step2_preprocess import make_canny, make_depth, make_openpose, make_normal
        
        # Générer tous les pass de ControlNet
        print("   🎨 Génération de tous les pass ControlNet...")
        
        # Pass 1: Canny (contours)
        try:
            # Paramètres adaptatifs selon la structure de scène
            if scene_structure == "aerial":
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
        
        # Pass 3: OpenPose (poses) - Skip pour scènes aériennes
        if scene_structure != "aerial":
            try:
                control_images["openpose"] = make_openpose(current_image, save_path="output/controlnet_openpose.png")
            except Exception as e:
                print(f"   ⚠️  Erreur OpenPose: {e}")
        else:
            print(f"   ⏭️  OpenPose désactivé pour scènes aériennes")
        
        # Pass 4: Normal (normales) - Skip pour scènes aériennes
        if scene_structure != "aerial":
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
            # Passer la configuration de prompt pour un meilleur filtrage
            segment_target = detect_segment_target(user_prompt, scene_type=scene_structure)
            print(f"   🎯 Cible détectée automatiquement: {get_target_description(segment_target)}")
        else:
            print(f"   🎯 Cible spécifiée manuellement: {get_target_description(segment_target)}")
        
        mask = segment_target_region(
            image=current_image,
            target=segment_target,
            method=segment_method,
            scene_type=scene_structure,  # Passer la structure de scène
            dilate=SEGMENT_DILATE,
            feather=SEGMENT_FEATHER,
            save_path="output/segmentation_mask.png"
        )
        
        # Pour les scènes aériennes, charger les métadonnées des éléments détectés
        if scene_structure == "aerial":
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
                prompt_config=prompt_config,  # Nouvelle configuration modulaire
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
                prompt_config=prompt_config,  # Nouvelle configuration modulaire
                width=width,
                height=height,
                seed=SEED,
                aerial_elements=aerial_elements  # Passer les éléments aériens
            )
            
        else:
            # Mode CONTROLNET classique (sans masque)
            print("   🎯 Mode: CONTROLNET classique")
            from models.sdxl import load_sdxl
            from steps.step3_generate import generate_with_sdxl, generate_aerial_multipass
            
            pipe, refiner = load_sdxl(
                SDXL_MODEL, 
                CONTROLNET_MODEL, 
                enable_refiner and USE_REFINER
            )
            
            print("\n🎭 Étape 6: Génération SDXL")
            control_img = control_images.get("depth") if control_images else None
            print(f"   🎛️  ControlNet utilisé: {'Depth' if control_img else 'None'}")
            
            # === SCÈNES AÉRIENNES: 3 PASSES ===
            if scene_structure == "aerial" and aerial_elements:
                print("   🚁 Scène aérienne détectée → Génération multi-pass (3 passes)")
                print(f"   📋 Éléments détectés: {', '.join(aerial_elements)}")
                current_image = generate_aerial_multipass(
                    image=current_image,
                    control_images=control_images,
                    pipe=pipe,
                    refiner=refiner if enable_refiner else None,
                    user_prompt=user_prompt,
                    width=width,
                    height=height,
                    seed=SEED,
                    aerial_elements=aerial_elements,
                    prompt_config=prompt_config  # Nouvelle configuration modulaire
                )
            else:
                # === AUTRES SCÈNES: 1 PASSE CLASSIQUE ===
                # Strength ajusté pour scènes aériennes
                strength_value = 0.65 if scene_structure == "aerial" else 0.20
                
                current_image = generate_with_sdxl(
                    image=current_image,
                    control_image=control_img,
                    pipe=pipe,
                    refiner=refiner if enable_refiner else None,
                    prompt_config=prompt_config,  # Nouvelle configuration modulaire
                    width=width,
                    height=height,
                    seed=SEED,
                    strength=strength_value,  # Denoise +0.45 pour AERIAL (0.65 total)
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
        "prompt_config": prompt_config,
        "mask": mask,
        "cloudinary_url": cloudinary_url,
        "dimensions": final_image.size,
        "last_step_executed": last_step,
        "steps_executed": {
            "controlnet": enable_controlnet,
            "segmentation": enable_segmentation,
            "sdxl": enable_sdxl,
            "refiner": enable_refiner,
            "upscaler": enable_upscaler,
            "upload": enable_upload
        }
    }
