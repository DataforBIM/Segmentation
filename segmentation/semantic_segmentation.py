# =====================================================
# ÉTAPE 3: SEGMENTATION SÉMANTIQUE
# =====================================================
# Utilise OneFormer/Mask2Former/SegFormer pour comprendre la scène globale
# Chaque pixel est classifié en catégorie

import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

# Cache global des modèles
_segformer_model = None
_segformer_processor = None
_oneformer_model = None
_oneformer_processor = None

# Modèle par défaut
DEFAULT_MODEL = "oneformer"  # "oneformer", "mask2former", "segformer"


@dataclass
class SemanticMap:
    """Map sémantique complète de l'image"""
    
    # Masques individuels par classe
    masks: dict = field(default_factory=dict)  # {"floor": PIL.Image, "wall": PIL.Image, ...}
    
    # Map de classes brute (chaque pixel = class_id)
    class_map: Optional[np.ndarray] = None
    
    # Confiances par classe
    confidences: dict = field(default_factory=dict)
    
    # Dimensions originales
    size: tuple = (0, 0)
    
    # Classes détectées
    detected_classes: list = field(default_factory=list)


# =====================================================
# MAPPING DES CLASSES ADE20K
# =====================================================
# SegFormer utilise ADE20K (150 classes)

ADE20K_CLASSES = {
    # Surfaces architecturales
    "floor": [3],  # floor, flooring
    "wall": [0],   # wall
    "ceiling": [5],  # ceiling
    "carpet": [28],  # rug, carpet
    
    # Ouvertures
    "window": [8],  # window
    "door": [14],   # door
    
    # Mobilier
    "furniture": [7, 10, 15, 19, 24, 33, 36, 45, 57, 62, 65, 75],
    # 7=bed, 10=table, 15=sofa, 19=chair, 24=cabinet, 33=armchair, etc.
    "sofa": [15],
    "table": [10],
    "chair": [19],
    "bed": [7],
    "cabinet": [24, 35],
    
    # Personnes et êtres vivants
    "person": [12],  # person
    
    # Extérieur
    "building": [1],  # building, edifice
    "sky": [2],       # sky
    "tree": [4],      # tree
    "road": [6],      # road, route
    "grass": [9],     # grass
    "ground": [6, 9, 11, 13],  # ✨ ground = road + grass + sidewalk + earth/sand
    "sidewalk": [11], # sidewalk, pavement
    "vegetation": [4, 9, 17, 66],  # tree, grass, plant, flower
    "flowers": [66],  # ✨ flower (séparé de vegetation)
    "plant": [17],    # ✨ plant (séparé)
    "car": [20],      # car
    
    # Architecture
    "roof": [51],     # roof (approximatif)
    "stairs": [53],   # stairs
    "column": [42],   # column, pillar
    
    # Objets
    "lamp": [36, 82], # lamp, light
    "object": [39, 64, 67, 74],  # objets génériques
}

# Mapping inversé: class_id → nom
CLASS_ID_TO_NAME = {}
for name, ids in ADE20K_CLASSES.items():
    for class_id in ids:
        CLASS_ID_TO_NAME[class_id] = name


def load_oneformer():
    """
    Charge OneFormer pour la segmentation panoptique
    Utilise le modèle ADE20K Swin Large
    """
    global _oneformer_model, _oneformer_processor
    
    if _oneformer_model is not None:
        print("   ♻️  OneFormer déjà chargé (cache)")
        return _oneformer_model, _oneformer_processor
    
    print("   🔷 Chargement de OneFormer...")
    
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    
    _oneformer_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    )
    _oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    ).to("cuda")
    
    print("   ✅ OneFormer chargé (shi-labs/oneformer_ade20k_swin_large)")
    
    return _oneformer_model, _oneformer_processor


def unload_oneformer():
    """Décharge OneFormer de la mémoire"""
    global _oneformer_model, _oneformer_processor
    
    if _oneformer_model is not None:
        del _oneformer_model
        del _oneformer_processor
        _oneformer_model = None
        _oneformer_processor = None
        torch.cuda.empty_cache()
        print("   🗑️  OneFormer déchargé")


# =====================================================
# CHARGEMENT DU MODÈLE
# =====================================================

def load_segformer():
    """
    Charge SegFormer pour la segmentation sémantique
    Utilise le modèle ADE20K 640x640
    """
    global _segformer_model, _segformer_processor
    
    if _segformer_model is not None:
        print("   ♻️  SegFormer déjà chargé (cache)")
        return _segformer_model, _segformer_processor
    
    print("   🧠 Chargement de SegFormer...")
    
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    
    _segformer_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    _segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        torch_dtype=torch.float32
    ).to("cuda")
    
    print("   ✅ SegFormer chargé (nvidia/segformer-b5-finetuned-ade-640-640)")
    
    return _segformer_model, _segformer_processor


def unload_segformer():
    """Décharge SegFormer de la mémoire"""
    global _segformer_model, _segformer_processor
    
    if _segformer_model is not None:
        del _segformer_model
        del _segformer_processor
        _segformer_model = None
        _segformer_processor = None
        torch.cuda.empty_cache()
        print("   🗑️  SegFormer déchargé")


# =====================================================
# SEGMENTATION SÉMANTIQUE PRINCIPALE
# =====================================================

def semantic_segment(
    image: Image.Image,
    targets: list[str] = None,
    save_path: str = None,
    model_type: str = None
) -> SemanticMap:
    """
    Segmente l'image en classes sémantiques
    
    Args:
        image: Image PIL d'entrée
        targets: Liste optionnelle des cibles à extraire
        save_path: Chemin pour sauvegarder la visualisation
        model_type: "oneformer", "mask2former", "segformer" (défaut: oneformer)
    
    Returns:
        SemanticMap avec tous les masques détectés
    
    Example:
        >>> sem_map = semantic_segment(image, targets=["floor", "wall"])
        >>> floor_mask = sem_map.masks["floor"]
    """
    
    if model_type is None:
        model_type = DEFAULT_MODEL
    
    if model_type == "oneformer":
        return _segment_with_oneformer(image, targets, save_path)
    elif model_type == "segformer":
        return _segment_with_segformer(image, targets, save_path)
    else:
        raise ValueError(f"Modèle non supporté: {model_type}")


def _segment_with_oneformer(
    image: Image.Image,
    targets: list[str] = None,
    save_path: str = None
) -> SemanticMap:
    """Segmentation avec OneFormer (panoptique)"""
    
    print("   🔷 Segmentation sémantique (OneFormer Panoptic)...")
    
    model, processor = load_oneformer()
    
    # Préparation pour segmentation panoptique
    task = "panoptic"
    inputs = processor(images=image, task_inputs=[task], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-processing panoptique
    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]
    
    # Extraire segmentation_map et segments_info
    segmentation_map = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]
    
    # Créer les masques par classe
    masks = {}
    detected_classes = []
    confidences = {}
    
    id2label = model.config.id2label
    
    for segment in segments_info:
        segment_id = segment["id"]
        label_id = segment["label_id"]
        label = id2label[label_id].lower()
        score = segment.get("score", 1.0)
        
        # Créer le masque binaire pour ce segment
        mask_array = (segmentation_map == segment_id).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_array, mode="L")
        
        # Fusionner si la classe existe déjà
        if label in masks:
            existing = np.array(masks[label])
            combined = np.maximum(existing, mask_array)
            masks[label] = Image.fromarray(combined, mode="L")
        else:
            masks[label] = mask_pil
            detected_classes.append(label)
        
        # Garder le meilleur score
        if label not in confidences or score > confidences[label]:
            confidences[label] = score
    
    print(f"   ✅ {len(detected_classes)} classes détectées: {detected_classes[:10]}...")
    
    return SemanticMap(
        masks=masks,
        class_map=segmentation_map,
        confidences=confidences,
        size=image.size,
        detected_classes=detected_classes
    )


def _segment_with_segformer(
    image: Image.Image,
    targets: list[str] = None,
    save_path: str = None
) -> SemanticMap:
    """Segmentation avec SegFormer (sémantique pure)"""
    import torch.nn.functional as F
    
    print("   🎯 Segmentation sémantique (SegFormer)...")
    
    model, processor = load_segformer()
    
    # Préparation
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.float32) for k, v in inputs.items()}
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Redimensionner à la taille originale
    upsampled = F.interpolate(
        logits,
        size=(image.size[1], image.size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False
    )
    
    # Obtenir les classes prédites
    class_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    
    # Obtenir les probabilités pour la confiance
    probs = torch.softmax(upsampled, dim=1).squeeze().cpu().numpy()
    
    # Créer les masques pour chaque classe
    masks = {}
    confidences = {}
    detected_classes = []
    
    # Si targets spécifiés, ne garder que ceux-là
    targets_to_process = targets if targets else list(ADE20K_CLASSES.keys())
    
    for class_name in targets_to_process:
        if class_name not in ADE20K_CLASSES:
            continue
        
        class_ids = ADE20K_CLASSES[class_name]
        
        # Créer le masque binaire
        mask = np.isin(class_map, class_ids).astype(np.uint8) * 255
        
        # Vérifier si la classe est présente
        coverage = np.sum(mask > 0) / mask.size
        
        if coverage > 0.001:  # Au moins 0.1% de couverture
            masks[class_name] = Image.fromarray(mask, mode="L")
            
            # Calculer la confiance moyenne pour cette classe
            class_probs = np.zeros_like(mask, dtype=np.float32)
            for class_id in class_ids:
                if class_id < probs.shape[0]:
                    class_probs = np.maximum(class_probs, probs[class_id])
            conf = np.mean(class_probs[mask > 0]) if np.any(mask > 0) else 0
            confidences[class_name] = float(conf)
            
            detected_classes.append(class_name)
            print(f"      ✓ {class_name}: {coverage*100:.1f}% (conf: {conf:.2f})")
    
    # Sauvegarder la visualisation si demandé
    if save_path:
        _save_semantic_visualization(image, masks, save_path)
    
    print(f"   ✅ {len(detected_classes)} classes détectées")
    
    return SemanticMap(
        masks=masks,
        class_map=class_map,
        confidences=confidences,
        size=image.size,
        detected_classes=detected_classes
    )


# =====================================================
# FONCTIONS D'EXTRACTION DE MASQUES
# =====================================================

def get_mask(semantic_map: SemanticMap, class_name: str) -> Optional[Image.Image]:
    """Récupère le masque d'une classe spécifique"""
    return semantic_map.masks.get(class_name)


def get_combined_mask(
    semantic_map: SemanticMap,
    class_names: list[str],
    mode: str = "union"
) -> Image.Image:
    """
    Combine plusieurs masques en un seul
    
    Args:
        semantic_map: Map sémantique
        class_names: Liste des classes à combiner
        mode: "union" (OR) ou "intersection" (AND)
    
    Returns:
        Masque combiné
    """
    
    # Mapping des alias vers les vraies classes ADE20K
    CLASS_ALIASES = {
        "ground": ["grass", "field", "earth", "sand", "dirt", "soil", "path", "pavement"],
        "vegetation": ["tree", "plant", "flower", "grass", "bush", "palm"],
        "furniture": ["chair", "table", "sofa", "bed", "cabinet", "desk", "shelf"],
        "object": ["box", "bag", "bottle", "book"],
    }
    
    combined = None
    found_classes = []
    
    for class_name in class_names:
        # Chercher la classe directement
        mask = semantic_map.masks.get(class_name)
        
        if mask is not None:
            # Trouvé directement
            found_classes.append(class_name)
            mask_array = np.array(mask)
            
            if combined is None:
                combined = mask_array
            else:
                if mode == "union":
                    combined = np.maximum(combined, mask_array)
                elif mode == "intersection":
                    combined = np.minimum(combined, mask_array)
        
        # Si pas trouvé directement, chercher dans les alias
        elif class_name in CLASS_ALIASES:
            for alias in CLASS_ALIASES[class_name]:
                alias_mask = semantic_map.masks.get(alias)
                if alias_mask is not None:
                    found_classes.append(f"{class_name}→{alias}")
                    alias_array = np.array(alias_mask)
                    if combined is None:
                        combined = alias_array
                    else:
                        if mode == "union":
                            combined = np.maximum(combined, alias_array)
                        elif mode == "intersection":
                            combined = np.minimum(combined, alias_array)
    
    # Debug: afficher ce qui a été trouvé
    if found_classes:
        print(f"      🔍 Classes trouvées: {', '.join(found_classes)}")
    else:
        print(f"      ⚠️  Aucune classe trouvée parmi: {class_names}")
        print(f"      📋 Classes disponibles: {list(semantic_map.masks.keys())[:15]}")
    
    if combined is None:
        # Retourner un masque vide
        return Image.new("L", semantic_map.size, 0)
    
    return Image.fromarray(combined, mode="L")


def subtract_masks(
    base_mask: Image.Image,
    subtract_masks: list[Image.Image]
) -> Image.Image:
    """
    Soustrait des masques d'un masque de base
    
    Args:
        base_mask: Masque de base
        subtract_masks: Liste des masques à soustraire
    
    Returns:
        Masque résultant
    
    Example:
        >>> # Façade sans fenêtres
        >>> facade_clean = subtract_masks(facade_mask, [windows_mask, doors_mask])
    """
    result = np.array(base_mask)
    
    for mask in subtract_masks:
        if mask is not None:
            mask_array = np.array(mask)
            # Où mask est blanc, mettre result à noir
            result = np.where(mask_array > 127, 0, result)
    
    return Image.fromarray(result.astype(np.uint8), mode="L")


def prepare_facade_masks(
    semantic_map: SemanticMap,
    image_size: tuple
) -> dict:
    """
    Prépare les masques de façade avec protection des ouvertures
    
    Sépare proprement:
    - Façade (upper + middle + lower) SANS fenêtres/portes
    - Fenêtres/portes (protégées)
    
    Args:
        semantic_map: Map sémantique
        image_size: Taille de l'image
    
    Returns:
        Dict avec:
        {
            "facade_full": masque complet de la façade (avec ouvertures)
            "facade_clean": façade SANS fenêtres/portes (pour modification)
            "windows": fenêtres (protégées)
            "doors": portes (protégées)
            "protected": fenêtres + portes combinées
            "facade_upper_clean": tiers supérieur sans ouvertures
            "facade_middle_clean": tiers milieu sans ouvertures
            "facade_lower_clean": tiers inférieur sans ouvertures
        }
    """
    
    print("   🔧 Préparation des masques de façade...")
    
    # 1. Extraire les masques architecturaux
    arch_masks = extract_architectural_masks(semantic_map, image_size, split_facade=True)
    
    result = {}
    
    # 2. Combiner toutes les zones de façade
    facade_zones = []
    for zone in ["facade_upper", "facade_middle", "facade_lower"]:
        if zone in arch_masks:
            facade_zones.append(np.array(arch_masks[zone]))
    
    if facade_zones:
        facade_full = np.maximum.reduce(facade_zones)
        result["facade_full"] = Image.fromarray(facade_full.astype(np.uint8), mode="L")
    else:
        result["facade_full"] = None
    
    # 3. Extraire les ouvertures (fenêtres + portes)
    windows_mask = arch_masks.get("windows")
    doors_mask = arch_masks.get("doors")
    
    result["windows"] = windows_mask
    result["doors"] = doors_mask
    
    # 4. Combiner fenêtres + portes = protected
    protected_masks = []
    if windows_mask:
        protected_masks.append(np.array(windows_mask))
    if doors_mask:
        protected_masks.append(np.array(doors_mask))
    
    if protected_masks:
        protected_combined = np.maximum.reduce(protected_masks)
        result["protected"] = Image.fromarray(protected_combined.astype(np.uint8), mode="L")
    else:
        result["protected"] = None
    
    # 5. Soustraire les ouvertures de la façade complète
    if result["facade_full"] and result["protected"]:
        result["facade_clean"] = subtract_masks(
            result["facade_full"],
            [result["protected"]]
        )
        print("   ✅ Façade nettoyée (fenêtres/portes retirées)")
    else:
        result["facade_clean"] = result["facade_full"]
    
    # 6. Nettoyer chaque zone individuellement
    for zone in ["facade_upper", "facade_middle", "facade_lower"]:
        if zone in arch_masks and result["protected"]:
            clean_zone = subtract_masks(
                arch_masks[zone],
                [result["protected"]]
            )
            result[f"{zone}_clean"] = clean_zone
        elif zone in arch_masks:
            result[f"{zone}_clean"] = arch_masks[zone]
    
    # 7. Calculer les statistiques
    if result["facade_full"]:
        full_coverage = np.sum(np.array(result["facade_full"]) > 0) / (image_size[0] * image_size[1])
        clean_coverage = np.sum(np.array(result["facade_clean"]) > 0) / (image_size[0] * image_size[1])
        protected_coverage = np.sum(np.array(result["protected"]) > 0) / (image_size[0] * image_size[1]) if result["protected"] else 0
        
        print(f"   📊 Façade complète: {full_coverage*100:.1f}%")
        print(f"   📊 Façade nettoyée: {clean_coverage*100:.1f}%")
        print(f"   📊 Ouvertures protégées: {protected_coverage*100:.1f}%")
    
    return result


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def _save_semantic_visualization(
    image: Image.Image,
    masks: dict,
    save_path: str
):
    """Sauvegarde une visualisation colorée des masques"""
    
    # Couleurs pour chaque classe
    colors = {
        "floor": (139, 69, 19),      # Marron
        "wall": (169, 169, 169),     # Gris
        "ceiling": (255, 255, 224),  # Beige clair
        "window": (135, 206, 235),   # Bleu ciel
        "door": (139, 90, 43),       # Brun
        "furniture": (255, 165, 0),  # Orange
        "sofa": (255, 99, 71),       # Rouge tomate
        "table": (210, 180, 140),    # Tan
        "chair": (188, 143, 143),    # Rose
        "person": (255, 0, 0),       # Rouge
        "building": (128, 128, 128), # Gris
        "sky": (135, 206, 250),      # Bleu clair
        "vegetation": (34, 139, 34), # Vert forêt
        "road": (105, 105, 105),     # Gris foncé
        "car": (0, 0, 255),          # Bleu
        "roof": (160, 82, 45),       # Sienna
    }
    
    # Créer l'overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    
    for class_name, mask in masks.items():
        color = colors.get(class_name, (128, 128, 128))
        color_with_alpha = color + (100,)  # Transparence
        
        mask_array = np.array(mask)
        colored = Image.new("RGBA", image.size, color_with_alpha)
        
        # Appliquer le masque
        mask_rgba = Image.fromarray(mask_array).convert("L")
        overlay = Image.composite(colored, overlay, mask_rgba)
    
    # Combiner avec l'image originale
    result = Image.alpha_composite(image.convert("RGBA"), overlay)
    result.convert("RGB").save(save_path)
    print(f"   💾 Visualisation sauvegardée: {save_path}")


def get_class_coverage(semantic_map: SemanticMap, class_name: str) -> float:
    """Retourne le pourcentage de couverture d'une classe"""
    
    mask = semantic_map.masks.get(class_name)
    if mask is None:
        return 0.0
    
    mask_array = np.array(mask)
    return np.sum(mask_array > 0) / mask_array.size


def list_detected_classes(semantic_map: SemanticMap) -> list[tuple[str, float]]:
    """Liste les classes détectées avec leur couverture"""
    
    results = []
    for class_name in semantic_map.detected_classes:
        coverage = get_class_coverage(semantic_map, class_name)
        results.append((class_name, coverage))
    
    # Trier par couverture décroissante
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


# =====================================================
# EXTRACTION ARCHITECTURALE AVANCÉE
# =====================================================

def extract_architectural_masks(
    semantic_map: SemanticMap,
    image_size: tuple,
    split_facade: bool = True
) -> dict:
    """
    Extrait des masques architecturaux spécialisés
    
    Args:
        semantic_map: Map sémantique
        image_size: Taille de l'image (width, height)
        split_facade: Diviser la façade en zones verticales
    
    Returns:
        Dict avec masques spécialisés:
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
    
    result = {}
    
    # Sky
    sky_classes = ["sky", "clouds"]
    sky_mask = _combine_classes(semantic_map, sky_classes)
    if sky_mask:
        result["sky"] = sky_mask
    
    # Building/Facade
    building_classes = ["building", "edifice", "house", "facade", "wall-brick", "wall-stone"]
    building_mask = _combine_classes(semantic_map, building_classes)
    
    if building_mask and split_facade:
        # Diviser la façade en zones verticales
        facade_zones = _split_facade_vertically(building_mask, num_zones=3)
        result.update(facade_zones)
    elif building_mask:
        result["facade"] = building_mask
    
    # Windows
    window_classes = ["window", "windowpane", "glass"]
    windows_mask = _combine_classes(semantic_map, window_classes)
    if windows_mask:
        result["windows"] = windows_mask
    
    # Doors
    door_classes = ["door", "door-stuff"]
    doors_mask = _combine_classes(semantic_map, door_classes)
    if doors_mask:
        result["doors"] = doors_mask
    
    # Roof
    roof_classes = ["roof", "canopy", "awning"]
    roof_mask = _combine_classes(semantic_map, roof_classes)
    if roof_mask:
        result["roof"] = roof_mask
    
    # Vegetation
    vegetation_classes = ["tree", "plant", "grass", "flower", "bush", "palm"]
    vegetation_mask = _combine_classes(semantic_map, vegetation_classes)
    if vegetation_mask:
        result["vegetation"] = vegetation_mask
    
    # Ground
    ground_classes = ["road", "sidewalk", "ground", "pavement", "path", "earth"]
    ground_mask = _combine_classes(semantic_map, ground_classes)
    if ground_mask:
        result["ground"] = ground_mask
    
    return result


def _combine_classes(semantic_map: SemanticMap, class_names: list[str]) -> Optional[Image.Image]:
    """Combine plusieurs classes en un masque unique"""
    
    masks_to_combine = []
    for class_name in class_names:
        mask = semantic_map.masks.get(class_name)
        if mask is not None:
            masks_to_combine.append(np.array(mask))
    
    if not masks_to_combine:
        return None
    
    # Union de tous les masques
    combined = masks_to_combine[0]
    for mask in masks_to_combine[1:]:
        combined = np.maximum(combined, mask)
    
    return Image.fromarray(combined, mode="L")


def _split_facade_vertically(
    building_mask: Image.Image,
    num_zones: int = 3
) -> dict:
    """
    Divise un masque de bâtiment en zones verticales
    
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
        
        if np.any(zone_mask > 0):
            zones[name] = Image.fromarray(zone_mask, mode="L")
    
    return zones


# =====================================================
# PASSE 3: RAFFINEMENT SAM2
# =====================================================

def sample_points_from_mask(
    mask: Image.Image,
    num_points: int = 10,
    strategy: str = "grid"
) -> list:
    """
    Échantillonne des points depuis un masque pour SAM2
    
    Args:
        mask: Masque PIL
        num_points: Nombre de points à échantillonner
        strategy: "grid", "random", "edges", "center"
    
    Returns:
        Liste de tuples (x, y)
    """
    
    mask_array = np.array(mask)
    
    # Trouver tous les pixels blancs
    white_pixels = np.argwhere(mask_array > 127)
    
    if len(white_pixels) == 0:
        return []
    
    points = []
    
    if strategy == "grid":
        # Grille régulière dans la zone masquée
        y_coords, x_coords = white_pixels[:, 0], white_pixels[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        grid_size = int(np.sqrt(num_points)) + 1
        collected = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if collected >= num_points:
                    break
                    
                y = int(y_min + (y_max - y_min) * i / max(1, grid_size - 1))
                x = int(x_min + (x_max - x_min) * j / max(1, grid_size - 1))
                
                # Vérifier que le point est dans le masque
                if mask_array[y, x] > 127:
                    points.append((x, y))
                    collected += 1
                else:
                    # Trouver le pixel blanc le plus proche
                    distances = np.sqrt((white_pixels[:, 0] - y)**2 + (white_pixels[:, 1] - x)**2)
                    closest_idx = np.argmin(distances)
                    closest_y, closest_x = white_pixels[closest_idx]
                    points.append((closest_x, closest_y))
                    collected += 1
    
    elif strategy == "random":
        # Points aléatoires
        indices = np.random.choice(len(white_pixels), min(num_points, len(white_pixels)), replace=False)
        for idx in indices:
            y, x = white_pixels[idx]
            points.append((x, y))
    
    elif strategy == "center":
        # Centre de masse + quelques points autour
        from scipy import ndimage
        cy, cx = ndimage.center_of_mass(mask_array > 127)
        points.append((int(cx), int(cy)))
        
        # Ajouter des points autour du centre
        radius = min(mask_array.shape) // 10
        for angle in np.linspace(0, 2*np.pi, num_points-1, endpoint=False):
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            if 0 <= y < mask_array.shape[0] and 0 <= x < mask_array.shape[1]:
                if mask_array[y, x] > 127:
                    points.append((x, y))
    
    elif strategy == "edges":
        # Points sur les bords
        from scipy import ndimage
        
        # Détecter les bords
        edges = ndimage.sobel(mask_array.astype(float))
        edge_pixels = np.argwhere(edges > 0)
        
        if len(edge_pixels) > 0:
            indices = np.random.choice(len(edge_pixels), min(num_points, len(edge_pixels)), replace=False)
            for idx in indices:
                y, x = edge_pixels[idx]
                points.append((x, y))
    
    return points[:num_points]


def refine_mask_with_sam2(
    image: Image.Image,
    semantic_mask: Image.Image,
    sam2_model=None,
    sam2_processor=None,
    num_points: int = 10,
    strategy: str = "grid"
) -> Image.Image:
    """
    PASSE 3: Raffinement par objet avec SAM2
    
    Affine les bords de façade détectés par OneFormer:
    - Bords de façade précis
    - Découpes fines autour des menuiseries
    - Nettoyage des zones ambiguës (ombres, plantes proches)
    
    Args:
        image: Image originale
        semantic_mask: Masque sémantique de OneFormer
        sam2_model: Modèle SAM2 (chargé si None)
        sam2_processor: Processeur SAM2 (chargé si None)
        num_points: Nombre de points à échantillonner
        strategy: Stratégie d'échantillonnage
    
    Returns:
        Masque raffiné par SAM2
    """
    
    print(f"   🎯 PASSE 3: Raffinement SAM2...")
    
    # 1. Échantillonner des points depuis le masque sémantique
    points = sample_points_from_mask(semantic_mask, num_points, strategy)
    
    if not points:
        print(f"      ⚠️ Aucun point échantillonné, masque vide")
        return semantic_mask
    
    print(f"      📍 {len(points)} points échantillonnés (stratégie: {strategy})")
    
    # 2. Charger SAM2 si nécessaire
    if sam2_model is None or sam2_processor is None:
        from models.sam2 import load_sam2
        sam2_model, sam2_processor = load_sam2()
    
    # 3. Préparer les inputs pour SAM2
    # Format: [image_batch, object_batch, points, coords]
    # Convertir [(x,y), ...] -> [[[[x,y], ...]]]
    nested_points = [[[[int(x), int(y)] for x, y in points]]]
    
    inputs = sam2_processor(
        image,
        input_points=nested_points,
        return_tensors="pt"
    )
    
    # Déplacer sur GPU
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to("cuda")
    
    # 4. Générer le masque
    with torch.no_grad():
        outputs = sam2_model(**inputs)
    
    # 5. Extraire le masque
    masks = outputs.pred_masks.squeeze().cpu().numpy()
    
    # Prendre le premier masque si plusieurs
    if masks.ndim == 3:
        refined_mask = masks[0]
    else:
        refined_mask = masks
    
    # 6. Convertir en PIL
    refined_mask_pil = Image.fromarray((refined_mask > 0.5).astype(np.uint8) * 255, mode="L")
    
    # Redimensionner si nécessaire
    if refined_mask_pil.size != semantic_mask.size:
        refined_mask_pil = refined_mask_pil.resize(semantic_mask.size, Image.BILINEAR)
    
    # 7. Statistiques
    semantic_coverage = np.sum(np.array(semantic_mask) > 127) / (semantic_mask.size[0] * semantic_mask.size[1])
    refined_coverage = np.sum(np.array(refined_mask_pil) > 127) / (refined_mask_pil.size[0] * refined_mask_pil.size[1])
    
    print(f"      📊 Sémantique: {semantic_coverage*100:.1f}% → Raffiné: {refined_coverage*100:.1f}%")
    
    return refined_mask_pil

