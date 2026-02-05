# =====================================================
# SEGMENTATION SÉMANTIQUE AVANCÉE - OneFormer / Mask2Former
# =====================================================
# Segmentation panoptique pour des détails architecturaux fins

import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class PanopticSegmentation:
    """Résultat de segmentation panoptique"""
    
    # Masques par instance et classe
    semantic_masks: Dict[str, Image.Image] = field(default_factory=dict)
    
    # Instances individuelles
    instances: List[dict] = field(default_factory=list)
    
    # Map sémantique (class_id par pixel)
    semantic_map: Optional[np.ndarray] = None
    
    # Map d'instances (instance_id par pixel)
    instance_map: Optional[np.ndarray] = None
    
    # Classes détectées
    detected_classes: List[str] = field(default_factory=list)
    
    # Confiances
    confidences: Dict[str, float] = field(default_factory=dict)


# =====================================================
# MAPPING ARCHITECTURAL SPÉCIALISÉ
# =====================================================

# Mapping pour détecter des sous-parties de bâtiments
ARCHITECTURAL_CLASSES = {
    # Façades (nécessite post-processing spatial)
    "facade_upper": {"base_class": "building", "position": "upper_third"},
    "facade_middle": {"base_class": "building", "position": "middle_third"},
    "facade_lower": {"base_class": "building", "position": "lower_third"},
    
    # Ouvertures
    "windows": {"base_class": "window"},
    "doors": {"base_class": "door"},
    
    # Structure
    "roof": {"base_class": "roof"},
    "balcony": {"base_class": "balcony"},
    "stairs": {"base_class": "stairs"},
    
    # Environnement
    "vegetation": {"base_class": ["tree", "plant", "grass", "flower"]},
    "flowers": {"base_class": ["flower"]},  # ✨ NOUVEAU: fleurs spécifiques
    "plant": {"base_class": ["plant"]},  # ✨ NOUVEAU: plantes séparées
    "ground": {"base_class": ["road", "sidewalk", "grass", "ground"]},
    "sky": {"base_class": "sky"},
}


# =====================================================
# SEGMENTATION AVEC OneFormer
# =====================================================

def segment_with_oneformer(
    image: Image.Image,
    task: str = "panoptic",  # "semantic", "instance", "panoptic"
    confidence_threshold: float = 0.5
) -> PanopticSegmentation:
    """
    Segmentation panoptique avec OneFormer
    
    Args:
        image: Image PIL
        task: Type de segmentation ("panoptic" recommandé)
        confidence_threshold: Seuil de confiance
    
    Returns:
        PanopticSegmentation avec tous les masques
    """
    
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    
    print("   🔷 Chargement de OneFormer...")
    
    # Charger le modèle (ADE20K)
    processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    )
    model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    ).to("cuda")
    
    print(f"   ✅ OneFormer chargé (task: {task})")
    
    # Préparer l'image
    inputs = processor(images=image, task_inputs=[task], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-processing selon la tâche
    if task == "panoptic":
        result = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]
    elif task == "semantic":
        result = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]
    else:
        result = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]
    
    # Extraire les masques
    return _extract_panoptic_masks(result, image.size, model.config.id2label)


def _extract_panoptic_masks(
    result: dict,
    image_size: tuple,
    id2label: dict
) -> PanopticSegmentation:
    """Extrait les masques depuis le résultat OneFormer"""
    
    segmentation_map = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]
    
    semantic_masks = {}
    instances = []
    
    for segment in segments_info:
        segment_id = segment["id"]
        label_id = segment["label_id"]
        label = id2label[label_id]
        score = segment.get("score", 1.0)
        
        # Créer le masque pour ce segment
        mask = (segmentation_map == segment_id).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask, mode="L")
        
        # Ajouter aux masques sémantiques (fusionner si même classe)
        if label in semantic_masks:
            # Fusionner avec le masque existant
            existing = np.array(semantic_masks[label])
            combined = np.maximum(existing, mask)
            semantic_masks[label] = Image.fromarray(combined, mode="L")
        else:
            semantic_masks[label] = mask_pil
        
        # Ajouter l'instance
        instances.append({
            "label": label,
            "mask": mask_pil,
            "score": score,
            "area": np.sum(mask > 0)
        })
    
    return PanopticSegmentation(
        semantic_masks=semantic_masks,
        instances=instances,
        semantic_map=segmentation_map,
        detected_classes=list(semantic_masks.keys()),
        confidences={seg["label"]: seg.get("score", 1.0) for seg in segments_info}
    )


# =====================================================
# POST-PROCESSING: DIVISER LA FAÇADE EN ZONES
# =====================================================

def split_facade_vertically(
    building_mask: Image.Image,
    num_zones: int = 3
) -> Dict[str, Image.Image]:
    """
    Divise un masque de bâtiment verticalement
    
    Args:
        building_mask: Masque du bâtiment
        num_zones: Nombre de zones (3 = upper/middle/lower)
    
    Returns:
        Dict avec facade_upper, facade_middle, facade_lower
    """
    
    mask_array = np.array(building_mask)
    height, width = mask_array.shape
    
    # Trouver les limites verticales du bâtiment
    rows_with_building = np.any(mask_array > 0, axis=1)
    if not np.any(rows_with_building):
        return {}
    
    top = np.argmax(rows_with_building)
    bottom = height - np.argmax(rows_with_building[::-1])
    building_height = bottom - top
    
    # Diviser en zones
    zones = {}
    zone_height = building_height // num_zones
    
    zone_names = ["facade_upper", "facade_middle", "facade_lower"]
    
    for i, name in enumerate(zone_names[:num_zones]):
        zone_top = top + i * zone_height
        zone_bottom = zone_top + zone_height if i < num_zones - 1 else bottom
        
        # Créer le masque de zone
        zone_mask = np.zeros_like(mask_array)
        zone_mask[zone_top:zone_bottom, :] = mask_array[zone_top:zone_bottom, :]
        
        zones[name] = Image.fromarray(zone_mask, mode="L")
    
    return zones


# =====================================================
# FONCTION PRINCIPALE: SEGMENTATION ARCHITECTURALE
# =====================================================

def segment_architectural_scene(
    image: Image.Image,
    extract_facade_zones: bool = True
) -> Dict[str, Image.Image]:
    """
    Segmente une scène architecturale avec classes détaillées
    
    Args:
        image: Image PIL
        extract_facade_zones: Diviser la façade en zones upper/middle/lower
    
    Returns:
        Dict avec tous les masques:
        {
            "facade_upper": mask,
            "facade_lower": mask,
            "windows": mask,
            "doors": mask,
            "roof": mask,
            "vegetation": mask,
            "ground": mask,
            "sky": mask
        }
    """
    
    print("   🏛️  Segmentation architecturale avancée...")
    
    # 1. Segmentation panoptique avec OneFormer
    panoptic = segment_with_oneformer(image, task="panoptic")
    
    result = {}
    
    # 2. Extraire les classes de base
    for class_name, mask in panoptic.semantic_masks.items():
        
        # Sky
        if class_name in ["sky"]:
            result["sky"] = mask
        
        # Building / Facade
        elif class_name in ["building", "edifice", "house"]:
            if extract_facade_zones:
                # Diviser en zones verticales
                facade_zones = split_facade_vertically(mask)
                result.update(facade_zones)
            else:
                result["facade"] = mask
        
        # Windows
        elif class_name in ["window", "windowpane"]:
            result["windows"] = mask
        
        # Doors
        elif class_name in ["door"]:
            result["doors"] = mask
        
        # Roof
        elif class_name in ["roof"]:
            result["roof"] = mask
        
        # Vegetation
        elif class_name in ["tree", "plant", "grass", "flower", "bush"]:
            if "vegetation" in result:
                # Fusionner
                existing = np.array(result["vegetation"])
                new = np.array(mask)
                combined = np.maximum(existing, new)
                result["vegetation"] = Image.fromarray(combined, mode="L")
            else:
                result["vegetation"] = mask
        
        # Ground
        elif class_name in ["road", "sidewalk", "ground", "grass", "pavement"]:
            if "ground" in result:
                existing = np.array(result["ground"])
                new = np.array(mask)
                combined = np.maximum(existing, new)
                result["ground"] = Image.fromarray(combined, mode="L")
            else:
                result["ground"] = mask
    
    # 3. Afficher les résultats
    print(f"   ✅ Classes détectées:")
    for class_name, mask in result.items():
        coverage = np.sum(np.array(mask) > 0) / (mask.size[0] * mask.size[1])
        print(f"      - {class_name}: {coverage:.1%}")
    
    return result


# =====================================================
# EXEMPLE D'UTILISATION
# =====================================================

if __name__ == "__main__":
    from PIL import Image
    
    # Charger une image
    image = Image.open("input/building.jpg")
    
    # Segmentation architecturale
    semantic_masks = segment_architectural_scene(image)
    
    # Utiliser les masques
    facade_upper = semantic_masks.get("facade_upper")
    windows = semantic_masks.get("windows")
    
    # Sauvegarder
    for name, mask in semantic_masks.items():
        mask.save(f"output/semantic_{name}.png")
