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
    verbose: bool = True
) -> SegmentationResult:
    """
    Pipeline de segmentation intelligent
    
    ÉTAPES:
    1. Parse l'intention du prompt
    2. Résout les cibles (primary/protected/context)
    3. Segmentation sémantique (SegFormer)
    4. Segmentation instance (SAM2)
    5. Fusion des masques avec priorités
    6. Raffinement du masque
    7. Validation et auto-correction
    
    Args:
        image: Image PIL à segmenter
        user_prompt: Prompt utilisateur (ex: "change the floor to marble")
        sam2_predictor: Modèle SAM2 pré-chargé
        segformer_model: Modèle SegFormer pré-chargé
        segformer_processor: Processor SegFormer pré-chargé
        device: Device (cuda/cpu)
        auto_correct: Tenter l'auto-correction si masque invalide
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
        print(f"   Target: {intent.target}")
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
        print(f"   Primary classes: {target.primary_classes}")
        print(f"   Protected classes: {target.protected_classes}")
        print(f"   Context classes: {target.context_classes}")
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
        model=segformer_model,
        processor=segformer_processor,
        device=device
    )
    
    if verbose:
        detected_classes = list(semantic_map.class_masks.keys())[:10]
        print(f"   Classes détectées: {detected_classes}...")
        print()
    
    # =========================================================
    # ÉTAPE 4: SEGMENTATION D'INSTANCE (SAM2)
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 4: INSTANCE SEGMENTATION")
        print("━" * 40)
    
    # Obtenir le masque sémantique primaire
    primary_semantic_mask = get_combined_mask(
        semantic_map,
        target.primary_classes
    )
    
    # Appliquer SAM2 si disponible
    if sam2_predictor is not None and primary_semantic_mask is not None:
        
        instance_masks = instance_segment_from_semantic(
            predictor=sam2_predictor,
            image=image,
            semantic_mask=primary_semantic_mask,
            num_points=5
        )
        
        # Fusionner les instances
        if instance_masks:
            instance_mask = instance_masks[0]  # Meilleure instance
            for mask in instance_masks[1:]:
                instance_mask = Image.fromarray(
                    np.maximum(
                        np.array(instance_mask),
                        np.array(mask)
                    ),
                    mode="L"
                )
            primary_instance_mask = instance_mask
        else:
            primary_instance_mask = primary_semantic_mask
        
        if verbose:
            print(f"   ✓ SAM2: {len(instance_masks)} instances trouvées")
    else:
        primary_instance_mask = primary_semantic_mask
        
        if verbose:
            print(f"   ⚠️ SAM2 non disponible, utilisation masque sémantique")
    
    print()
    
    # =========================================================
    # ÉTAPE 5: FUSION DES MASQUES
    # =========================================================
    
    if verbose:
        print("━" * 40)
        print("ÉTAPE 5: MASK FUSION")
        print("━" * 40)
    
    # Créer les masques pour chaque layer
    protected_mask = get_combined_mask(
        semantic_map,
        target.protected_classes
    )
    
    context_mask = get_combined_mask(
        semantic_map,
        target.context_classes
    )
    
    # Fusionner avec les priorités
    mask_layers = fuse_masks(
        target_mask=primary_instance_mask,
        protected_mask=protected_mask,
        context_mask=context_mask,
        target_priority=target.priority
    )
    
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
        mask_layers.final_mask,
        image.size
    )
    
    refined_mask = refine_mask(
        mask_layers.final_mask,
        **refinement_params
    )
    
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
        target_mask=mask_layers.target_mask,
        protected_mask=mask_layers.protected_mask,
        context_mask=mask_layers.context_mask,
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
    refined = refine_mask(mask_layers.final_mask)
    
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
