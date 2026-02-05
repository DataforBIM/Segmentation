# =====================================================
# SEGMENTATION MODULE
# =====================================================
# Pipeline intelligent de segmentation style ChatGPT
#
# ARCHITECTURE:
# USER PROMPT → INTENT PARSER → TARGET RESOLVER →
# SEMANTIC SEGMENTATION → INSTANCE SEGMENTATION →
# MASK FUSION + PRIORITIES → REFINEMENT → VALIDATION

from .intent_parser import parse_intent, Intent
from .target_resolver import resolve_target, Target
from .semantic_segmentation import (
    semantic_segment,
    SemanticMap,
    get_combined_mask,
    subtract_masks,
    ADE20K_CLASSES,
    extract_architectural_masks,
    prepare_facade_masks,
    load_oneformer,
    load_segformer
)
from .instance_segmentation import (
    instance_segment_with_points,
    instance_segment_with_box,
    instance_segment_from_semantic
)
from .mask_fusion import fuse_masks, MaskLayers
from .mask_refinement import (
    refine_mask,
    dilate_mask,
    erode_mask,
    feather_mask,
    clean_mask_morphology,
    get_dynamic_refinement_params
)
from .validation import (
    validate_mask,
    auto_correct_mask,
    ValidationResult,
    ValidationStatus,
    validate_for_inpainting,
    validate_for_generation,
    compute_mask_metrics
)
from .pipeline import (
    segment_from_prompt,
    quick_segment,
    segment_element,
    load_segmentation_models,
    SegmentationResult
)


__all__ = [
    # === PIPELINE PRINCIPAL ===
    "segment_from_prompt",
    "quick_segment",
    "segment_element",
    "load_segmentation_models",
    "SegmentationResult",
    
    # === ÉTAPE 1: Intent Parser ===
    "parse_intent",
    "Intent",
    
    # === ÉTAPE 2: Target Resolver ===
    "resolve_target",
    "Target",
    
    # === ÉTAPE 3: Semantic Segmentation ===
    "semantic_segment",
    "SemanticMap",
    "get_combined_mask",
    "subtract_masks",
    "ADE20K_CLASSES",
    "extract_architectural_masks",
    "prepare_facade_masks",
    "load_oneformer",
    "load_segformer",
    
    # === ÉTAPE 4: Instance Segmentation ===
    "instance_segment_with_points",
    "instance_segment_with_box",
    "instance_segment_from_semantic",
    
    # === ÉTAPE 5: Mask Fusion ===
    "fuse_masks",
    "MaskLayers",
    
    # === ÉTAPE 6: Mask Refinement ===
    "refine_mask",
    "dilate_mask",
    "erode_mask",
    "feather_mask",
    "clean_mask_morphology",
    "get_dynamic_refinement_params",
    
    # === ÉTAPE 7: Validation ===
    "validate_mask",
    "auto_correct_mask",
    "ValidationResult",
    "ValidationStatus",
    "validate_for_inpainting",
    "validate_for_generation",
    "compute_mask_metrics",
]
