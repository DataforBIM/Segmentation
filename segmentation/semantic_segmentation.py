# =====================================================
# ÉTAPE 3: SEGMENTATION SÉMANTIQUE
# =====================================================
# Utilise SegFormer/OneFormer pour comprendre la scène globale
# Chaque pixel est classifié en catégorie

import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

# Cache global des modèles
_segformer_model = None
_segformer_processor = None


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
    "sidewalk": [11], # sidewalk, pavement
    "vegetation": [4, 9, 17, 66],  # tree, grass, plant, flower
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
    save_path: str = None
) -> SemanticMap:
    """
    Segmente l'image en classes sémantiques
    
    Args:
        image: Image PIL d'entrée
        targets: Liste optionnelle des cibles à extraire
        save_path: Chemin pour sauvegarder la visualisation
    
    Returns:
        SemanticMap avec tous les masques détectés
    
    Example:
        >>> sem_map = semantic_segment(image, targets=["floor", "wall"])
        >>> floor_mask = sem_map.masks["floor"]
    """
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
    combined = None
    
    for class_name in class_names:
        mask = semantic_map.masks.get(class_name)
        if mask is None:
            continue
        
        mask_array = np.array(mask)
        
        if combined is None:
            combined = mask_array
        else:
            if mode == "union":
                combined = np.maximum(combined, mask_array)
            elif mode == "intersection":
                combined = np.minimum(combined, mask_array)
    
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
    """
    result = np.array(base_mask)
    
    for mask in subtract_masks:
        if mask is not None:
            mask_array = np.array(mask)
            # Où mask est blanc, mettre result à noir
            result = np.where(mask_array > 127, 0, result)
    
    return Image.fromarray(result.astype(np.uint8), mode="L")


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
