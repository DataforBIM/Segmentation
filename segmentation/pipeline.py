# =====================================================
# SEGMENTATION PIPELINE - ORCHESTRATEUR PRINCIPAL
# =====================================================
# Pipeline intelligent style ChatGPT
# USER PROMPT → INTENT → TARGET → SEMANTIC → INSTANCE → FUSION → VALIDATION

import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .intent_parser import parse_intent, Intent
from .target_resolver import resolve_target, Target
from .semantic_segmentation import (
    semantic_segment,
    SemanticMap,
    get_combined_mask,
    subtract_masks
)
from .instance_segmentation import (
    instance_segment_with_points,
    instance_segment_from_semantic,
    sample_points_from_mask
)
from .mask_fusion import fuse_masks, MaskLayers
from .mask_refinement import refine_mask, get_dynamic_refinement_params
from .validation import (
    validate_mask,
    auto_correct_mask,
    ValidationResult,
    ValidationStatus
)


@dataclass
class SegmentationResult:
    """Résultat final de la segmentation"""
    
    # Masque final
    final_mask: Image.Image
    
    # Masques intermédiaires
    target_mask: Optional[Image.Image] = None
    protected_mask: Optional[Image.Image] = None
    context_mask: Optional[Image.Image] = None
    
    # ✨ NOUVEAU: Masques de transition pour blending
    transition_masks: Optional[Any] = None  # TransitionMasks
    
    # Métadonnées
    intent: Optional[Intent] = None
    target: Optional[Target] = None
    semantic_map: Optional[SemanticMap] = None
    validation: Optional[ValidationResult] = None
    
    # Stats
    coverage: float = 0.0
    processing_time: float = 0.0


# =====================================================
# PIPELINE PRINCIPAL
# =====================================================

def segment_from_prompt(
    image: Image.Image,
    user_prompt: str,
    sam2_predictor: Optional[Any] = None,
    segformer_model: Optional[Any] = None,
    segformer_processor: Optional[Any] = None,
    device: str = "cuda",
    auto_correct: bool = True,
    refine_target_with_sam2: bool = False,
    verbose: bool = True
) -> SegmentationResult:
    """
    Pipeline de segmentation intelligent
    
    ÉTAPES:
    1. Parse l'intention du prompt
    2. Résout les cibles (primary/protected/context)
    3. Segmentation sémantique (OneFormer/SegFormer)
    4. Segmentation instance (SAM2) - OPTIONNEL
    5. Fusion des masques avec priorités
    6. Raffinement du masque
    7. Validation et auto-correction
    
    Args:
        image: Image PIL à segmenter
        user_prompt: Prompt utilisateur (ex: "change the facade to white modern")
        sam2_predictor: Modèle SAM2 pré-chargé
        segformer_model: Modèle SegFormer pré-chargé
        segformer_processor: Processor SegFormer pré-chargé
        device: Device (cuda/cpu)
        auto_correct: Tenter l'auto-correction si masque invalide
        refine_target_with_sam2: Si True, raffine UNIQUEMENT le target avec SAM2
        verbose: Afficher les logs
    
    Returns:
        SegmentationResult avec le masque final et métadonnées
    """
    
    import time
    start_time = time.time()
    
    if verbose:
        print("=" * 60)
        print("🎯 SEGMENTATION PIPELINE")
        print("=" * 60)
        print(f"📝 Prompt: \"{user_prompt}\"")
        print(f"📐 Image: {image.size}")
        print()
    
    # =========================================================
    # ÉTAPE 1: PARSE L'INTENTION
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 1: INTENT PARSING")
        print("━" * 40)
    
    intent = parse_intent(user_prompt)
    
    if verbose:
        print(f"   Action: {intent.action}")
        print(f"   Action Type: {intent.action_type}")  # ✨ NOUVEAU
        print(f"   Target: {intent.target_hint}")
        if intent.action_type == "ADD":
            print(f"   Object to Add: {intent.object_to_add}")
            print(f"   Location: {intent.location}")
        print(f"   Material: {intent.material}")
        print(f"   Color: {intent.color}")
        print(f"   Style: {intent.style}")
        print()
    
    # =========================================================
    # ÉTAPE 2: RÉSOLUTION DES CIBLES
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 2: TARGET RESOLUTION")
        print("━" * 40)
    
    target = resolve_target(intent)
    
    if verbose:
        print(f"   Primary: {target.primary}")
        print(f"   Protected: {target.protected}")
        print(f"   Context: {target.context}")
        print()
    
    # =========================================================
    # ÉTAPE 3: SEGMENTATION SÉMANTIQUE
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 3: SEMANTIC SEGMENTATION")
        print("━" * 40)
    
    semantic_map = semantic_segment(
        image=image,
        model_type="oneformer"  # Utiliser OneFormer par défaut
    )
    
    if verbose:
        detected_classes = list(semantic_map.masks.keys())[:10]
        print(f"   ✅ Classes détectées: {', '.join(detected_classes)}")
        print()    
    # =========================================================
    # ÉTAPE 3.5: SPATIAL ZONE DETECTION (pour actions ADD)
    # =========================================================
    
    spatial_zone = None
    depth_map = None
    
    if intent.action_type == "ADD":
        if verbose:
            print("─" * 40)
            print("ÉTAPE 3.5: SPATIAL ZONE DETECTION (ADD)")
            print("─" * 40)
        
        # Générer depth map pour détection de zones
        try:
            from steps.step2_preprocess import make_depth
            depth_pil = make_depth(image, save_path="output/depth_map.png")
            depth_map = np.array(depth_pil)
            
            if verbose:
                print(f"   ✅ Depth map générée")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Depth map non disponible: {e}")
            depth_map = None
        
        # Détecter la zone spatiale
        from .spatial_zones import detect_spatial_zone
        
        zone_description = intent.location or "ground_foreground"
        
        spatial_zone = detect_spatial_zone(
            image=image,
            zone_description=zone_description,
            semantic_masks=semantic_map.masks,
            depth_map=depth_map
        )
        
        if verbose:
            from .spatial_zones import describe_zone
            print(f"   ✅ Zone détectée: {describe_zone(spatial_zone)}")
            
            # Sauvegarder preview de la zone
            from .spatial_zones import visualize_zone
            zone_preview = visualize_zone(image, spatial_zone, alpha=0.5)
            zone_preview.save("output/spatial_zone_preview.png")
            print(f"   ✅ Preview: output/spatial_zone_preview.png")
        
        print()    
    # =========================================================
    # ÉTAPE 4: INSTANCE SEGMENTATION (SAM2) - OPTIONNEL
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 4: INSTANCE SEGMENTATION (OPTIONNEL)")
        print("━" * 40)
    
    # Pour l'instant, on ne fait pas de segmentation d'instance séparée
    # SAM2 sera utilisé dans l'étape de fusion si refine_target_with_sam2=True
    
    if verbose:
        if refine_target_with_sam2:
            print(f"   ✅ Raffinement SAM2 activé (sera appliqué au target uniquement)")
        else:
            print(f"   ℹ️  Raffinement SAM2 désactivé")
    
    print()
    
    # =========================================================
    # ÉTAPE 5: FUSION DES MASQUES
    # =========================================================
    
    if verbose:
        print("━" * 40)
        if intent.action_type == "ADD":
            print("ÉTAPE 5: ADDITIVE MASK CREATION")
        else:
            print("ÉTAPE 5: MASK FUSION + SAM2 REFINEMENT")
        print("━" * 40)
    
    # NOUVEAU: Distinction ADD vs MODIFY
    if intent.action_type == "ADD" and spatial_zone:
        # MODE ADDITIF: Utiliser la zone spatiale comme masque
        if verbose:
            print(f"   ✨ Mode ADDITIF: Masque = zone d'accueil")
            print(f"   ❌ Pas de remplacement du contenu existant")
        
        # Le masque final = zone spatiale (avec protection intégrée)
        final_mask_pil = spatial_zone.mask
        
        # Créer mask_layers compatible
        from .mask_fusion import MaskLayers
        mask_layers = MaskLayers(
            target=final_mask_pil,
            protected=Image.new("L", image.size, 0),  # Pas de protection supplémentaire
            context=Image.new("L", image.size, 0),
            final=final_mask_pil
        )
        
        primary_semantic_mask = final_mask_pil
        
    else:
        # MODE CLASSIQUE: Fusionner avec les priorités
        # SAM2 sera appliqué au target uniquement si refine_target_with_sam2=True
        # Grounding DINO sera utilisé pour les ouvertures si manquantes
        mask_layers = fuse_masks(
            semantic_map=semantic_map,
            target=target,
            refine_target_with_sam2=refine_target_with_sam2,
            use_grounding_dino_for_protected=True,  # Approche hybride activée
            original_image=image
        )
        
        # Pour auto-correction
        primary_semantic_mask = get_combined_mask(semantic_map, target.primary)
    
    if verbose:
        print()
    
    # =========================================================
    # ÉTAPE 6: RAFFINEMENT DU MASQUE
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 6: MASK REFINEMENT")
        print("━" * 40)
    
    # Paramètres dynamiques
    refinement_params = get_dynamic_refinement_params(
        mask_layers.final,
        image.size
    )
    
    refined_mask = refine_mask(
        mask_layers.final,
        **refinement_params
    )
    
    if verbose:
        print()
    
    # =========================================================
    # ÉTAPE 6.5: CRÉATION DES MASQUES DE TRANSITION
    # =========================================================
    
    transition_masks = None
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 6.5: TRANSITION MASKS (BLENDING)")
        print("━" * 40)
    
    try:
        from .transition_masks import (
            create_transition_masks,
            compute_adaptive_transition_width,
            visualize_transition_masks,
            create_mask_comparison
        )
        
        # Calculer largeur adaptative
        transition_width = compute_adaptive_transition_width(
            refined_mask,
            image.size,
            base_width=12
        )
        
        # Créer masques de transition
        transition_masks = create_transition_masks(
            mask_core=refined_mask,
            transition_width=transition_width,
            gradient_type="cosine",  # Plus doux que linéaire
            adaptive_feather=True  # ✨ Feathering adaptatif basé sur aire du masque
        )
        
        if verbose:
            print(f"   ✅ Transition width: {transition_width}px")
            print(f"   ✅ Gradient type: cosine")
            
            # Sauvegarder visualisations
            visualize_transition_masks(
                image,
                transition_masks,
                save_path="output/transition_preview.png"
            )
            
            create_mask_comparison(
                transition_masks,
                save_path="output/transition_masks_comparison.png"
            )
            
            # Sauvegarder masques individuels
            transition_masks.core.save("output/mask_core.png")
            transition_masks.transition.save("output/mask_transition.png")
            transition_masks.combined.save("output/mask_combined.png")
            
            print(f"   💾 Preview: output/transition_preview.png")
            print(f"   💾 Comparison: output/transition_masks_comparison.png")
    
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Transition masks non créés: {e}")
        transition_masks = None
    
    if verbose:
        print()
    
    # =========================================================
    # ÉTAPE 7: VALIDATION
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 7: VALIDATION")
        print("━" * 40)
    
    validation_result = validate_mask(refined_mask)
    
    # Auto-correction si nécessaire
    if not validation_result.is_valid and auto_correct:
        validation_result = auto_correct_mask(
            mask=refined_mask,
            validation_result=validation_result,
            semantic_mask=primary_semantic_mask
        )
        refined_mask = validation_result.mask
    
    if verbose:
        print()
    
    # =========================================================
    # RÉSULTAT FINAL
    # =========================================================
    
    processing_time = time.time() - start_time
    
    # Calculer la couverture
    mask_array = np.array(refined_mask)
    coverage = np.sum(mask_array > 127) / mask_array.size
    
    if verbose:
        print("=" * 60)
        print("✅ SEGMENTATION TERMINÉE")
        print("=" * 60)
        print(f"   Couverture: {coverage:.1%}")
        print(f"   Temps: {processing_time:.2f}s")
        print(f"   Status: {validation_result.status.value}")
        print()
    
    return SegmentationResult(
        final_mask=refined_mask,
        target_mask=mask_layers.target,
        protected_mask=mask_layers.protected,
        context_mask=mask_layers.context,
        transition_masks=transition_masks,  # ✨ NOUVEAU
        intent=intent,
        target=target,
        semantic_map=semantic_map,
        validation=validation_result,
        coverage=coverage,
        processing_time=processing_time
    )


# =====================================================
# FONCTIONS SIMPLIFIÉES
# =====================================================

def quick_segment(
    image: Image.Image,
    target_classes: list,
    protected_classes: list = None,
    segformer_model: Optional[Any] = None,
    segformer_processor: Optional[Any] = None,
    device: str = "cuda"
) -> Image.Image:
    """
    Segmentation rapide sans parsing de prompt
    
    Args:
        image: Image à segmenter
        target_classes: Classes à cibler (ex: ["floor", "rug"])
        protected_classes: Classes à protéger (ex: ["person", "furniture"])
    
    Returns:
        Masque final
    """
    
    # Segmentation sémantique
    semantic_map = semantic_segment(
        image=image,
        model=segformer_model,
        processor=segformer_processor,
        device=device
    )
    
    # Masque target
    target_mask = get_combined_mask(semantic_map, target_classes)
    
    # Masque protected
    if protected_classes:
        protected_mask = get_combined_mask(semantic_map, protected_classes)
    else:
        protected_mask = None
    
    # Fusion
    mask_layers = fuse_masks(
        target_mask=target_mask,
        protected_mask=protected_mask
    )
    
    # Raffinement
    refined = refine_mask(mask_layers.final)
    
    return refined


def segment_element(
    image: Image.Image,
    element: str,
    segformer_model: Optional[Any] = None,
    segformer_processor: Optional[Any] = None,
    device: str = "cuda"
) -> Image.Image:
    """
    Segmente un élément spécifique
    
    Args:
        image: Image à segmenter
        element: Élément à cibler ("floor", "wall", "ceiling", etc.)
    
    Returns:
        Masque de l'élément
    """
    
    # Mapping simple
    ELEMENT_CLASSES = {
        "floor": ["floor", "rug", "carpet"],
        "wall": ["wall"],
        "ceiling": ["ceiling"],
        "furniture": ["sofa", "chair", "table", "bed", "cabinet"],
        "window": ["window", "windowpane"],
        "door": ["door"],
        "light": ["lamp", "chandelier", "light"],
        "plant": ["plant", "tree", "flower"],
        "art": ["painting", "poster"],
        "rug": ["rug", "carpet", "mat"]
    }
    
    target_classes = ELEMENT_CLASSES.get(element, [element])
    
    return quick_segment(
        image=image,
        target_classes=target_classes,
        segformer_model=segformer_model,
        segformer_processor=segformer_processor,
        device=device
    )


# =====================================================
# CHARGEMENT DES MODÈLES
# =====================================================

def load_segmentation_models(device: str = "cuda") -> dict:
    """
    Charge tous les modèles nécessaires pour la segmentation
    
    Returns:
        Dict avec segformer_model, segformer_processor, sam2_predictor
    """
    
    print("🔄 Chargement des modèles de segmentation...")
    
    models = {}
    
    # SegFormer
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        
        models["segformer_processor"] = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        )
        models["segformer_model"] = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640"
        ).to(device)
        
        print("   ✓ SegFormer chargé")
        
    except Exception as e:
        print(f"   ⚠️ Erreur SegFormer: {e}")
        models["segformer_model"] = None
        models["segformer_processor"] = None
    
    # SAM2
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        models["sam2_predictor"] = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large"
        )
        
        print("   ✓ SAM2 chargé")
        
    except Exception as e:
        print(f"   ⚠️ Erreur SAM2: {e}")
        models["sam2_predictor"] = None
    
    print("✅ Modèles de segmentation prêts")
    
    return models
