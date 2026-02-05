# =====================================================
# ÉTAPE 4: SEGMENTATION PAR INSTANCE (SAM2)
# =====================================================
# Utilise SAM2 pour segmenter des objets spécifiques
# Plus précis que la segmentation sémantique pour les instances

import torch
import numpy as np
from PIL import Image
from typing import Optional

# Cache global du modèle
_sam2_model = None
_sam2_processor = None


# =====================================================
# CHARGEMENT DU MODÈLE SAM2
# =====================================================

def load_sam2():
    """
    Charge SAM2 (Segment Anything Model 2) de Meta
    """
    global _sam2_model, _sam2_processor
    
    if _sam2_model is not None:
        print("   ♻️  SAM2 déjà chargé (cache)")
        return _sam2_model, _sam2_processor
    
    print("   🧠 Chargement de SAM2 (Meta)...")
    
    from transformers import Sam2Model, Sam2Processor
    
    _sam2_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    _sam2_model = Sam2Model.from_pretrained(
        "facebook/sam2-hiera-large",
        torch_dtype=torch.float16
    ).to("cuda")
    
    print("   ✅ SAM2 chargé (facebook/sam2-hiera-large)")
    
    return _sam2_model, _sam2_processor


def unload_sam2():
    """Décharge SAM2 de la mémoire"""
    global _sam2_model, _sam2_processor
    
    if _sam2_model is not None:
        del _sam2_model
        del _sam2_processor
        _sam2_model = None
        _sam2_processor = None
        torch.cuda.empty_cache()
        print("   🗑️  SAM2 déchargé")


# =====================================================
# SEGMENTATION PAR POINTS
# =====================================================

def instance_segment_with_points(
    image: Image.Image,
    points: list[tuple[int, int]],
    labels: list[int] = None,
    save_path: str = None
) -> Image.Image:
    """
    Segmente avec SAM2 en utilisant des points de référence
    
    Args:
        image: Image PIL d'entrée
        points: Liste de points (x, y) pour guider la segmentation
        labels: Liste de labels (1 = inclure, 0 = exclure)
        save_path: Chemin pour sauvegarder le masque
    
    Returns:
        Masque PIL de la région segmentée
    """
    print(f"   🎯 SAM2 avec {len(points)} points...")
    
    model, processor = load_sam2()
    
    if labels is None:
        labels = [1] * len(points)
    
    # Formatter pour SAM2
    points_formatted = [[list(p) for p in points]]
    labels_formatted = [labels]
    
    inputs = processor(
        images=image,
        input_points=[points_formatted],
        input_labels=[labels_formatted],
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement
    masks = outputs.pred_masks
    scores = outputs.iou_scores.cpu().numpy()
    
    # Prendre le meilleur masque
    scores_flat = scores.flatten()
    best_idx = scores_flat.argmax()
    best_score = scores_flat[best_idx]
    
    mask_tensor = masks[0, 0, best_idx % masks.shape[2]]
    mask = (mask_tensor > 0.5).float().cpu().numpy().astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    
    # Redimensionner si nécessaire
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.NEAREST)
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque SAM2 sauvegardé: {save_path}")
    
    print(f"   ✅ SAM2 terminé (IoU: {best_score:.3f})")
    
    return mask_image


# =====================================================
# SEGMENTATION PAR BOX
# =====================================================

def instance_segment_with_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    save_path: str = None
) -> Image.Image:
    """
    Segmente avec SAM2 en utilisant une bounding box
    
    Args:
        image: Image PIL d'entrée
        box: Bounding box (x1, y1, x2, y2)
        save_path: Chemin pour sauvegarder le masque
    
    Returns:
        Masque PIL de la région segmentée
    """
    print(f"   🎯 SAM2 avec box {box}...")
    
    model, processor = load_sam2()
    
    # Formatter la box pour SAM2
    input_boxes = [[[list(box)]]]
    
    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement
    masks = outputs.pred_masks
    scores = outputs.iou_scores.cpu().numpy()
    
    scores_flat = scores.flatten()
    best_idx = scores_flat.argmax()
    best_score = scores_flat[best_idx]
    
    mask_tensor = masks[0, 0, best_idx % masks.shape[2]]
    mask = (mask_tensor > 0.5).float().cpu().numpy().astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.NEAREST)
    
    if save_path:
        mask_image.save(save_path)
    
    print(f"   ✅ SAM2 (box) terminé (IoU: {best_score:.3f})")
    
    return mask_image


# =====================================================
# SEGMENTATION GUIDÉE PAR MASQUE SÉMANTIQUE
# =====================================================

def instance_segment_from_semantic(
    image: Image.Image,
    semantic_mask: Image.Image,
    num_points: int = 5,
    save_path: str = None
) -> Image.Image:
    """
    Utilise SAM2 pour affiner un masque sémantique
    
    Args:
        image: Image PIL d'entrée
        semantic_mask: Masque sémantique à affiner
        num_points: Nombre de points à échantillonner
        save_path: Chemin pour sauvegarder
    
    Returns:
        Masque affiné par SAM2
    """
    print(f"   🔄 Affinage SAM2 du masque sémantique...")
    
    # Échantillonner des points depuis le masque sémantique
    points = sample_points_from_mask(semantic_mask, num_points)
    
    if not points:
        print(f"   ⚠️  Pas de points trouvés dans le masque sémantique")
        return semantic_mask
    
    # Segmenter avec SAM2
    refined_mask = instance_segment_with_points(image, points, save_path=save_path)
    
    return refined_mask


def sample_points_from_mask(
    mask: Image.Image,
    num_points: int = 5,
    strategy: str = "grid"
) -> list[tuple[int, int]]:
    """
    Échantillonne des points depuis un masque
    
    Args:
        mask: Masque PIL
        num_points: Nombre de points à échantillonner
        strategy: "grid", "random", "center"
    
    Returns:
        Liste de points (x, y)
    """
    mask_array = np.array(mask)
    
    # Trouver les pixels blancs
    white_pixels = np.where(mask_array > 127)
    
    if len(white_pixels[0]) == 0:
        return []
    
    points = []
    
    if strategy == "center":
        # Point au centre du masque
        center_y = int(np.mean(white_pixels[0]))
        center_x = int(np.mean(white_pixels[1]))
        points.append((center_x, center_y))
    
    elif strategy == "grid":
        # Points sur une grille dans le masque
        min_y, max_y = white_pixels[0].min(), white_pixels[0].max()
        min_x, max_x = white_pixels[1].min(), white_pixels[1].max()
        
        # Créer une grille
        grid_size = int(np.sqrt(num_points))
        y_steps = np.linspace(min_y, max_y, grid_size + 2)[1:-1]
        x_steps = np.linspace(min_x, max_x, grid_size + 2)[1:-1]
        
        for y in y_steps:
            for x in x_steps:
                # Vérifier que le point est dans le masque
                y_int, x_int = int(y), int(x)
                if mask_array[y_int, x_int] > 127:
                    points.append((x_int, y_int))
                    if len(points) >= num_points:
                        break
            if len(points) >= num_points:
                break
    
    elif strategy == "random":
        # Points aléatoires dans le masque
        indices = np.random.choice(len(white_pixels[0]), min(num_points, len(white_pixels[0])), replace=False)
        for idx in indices:
            y = white_pixels[0][idx]
            x = white_pixels[1][idx]
            points.append((int(x), int(y)))
    
    return points[:num_points]


def get_bounding_box_from_mask(mask: Image.Image, padding: int = 10) -> tuple[int, int, int, int]:
    """
    Calcule la bounding box d'un masque
    
    Args:
        mask: Masque PIL
        padding: Padding autour de la box
    
    Returns:
        (x1, y1, x2, y2)
    """
    mask_array = np.array(mask)
    white_pixels = np.where(mask_array > 127)
    
    if len(white_pixels[0]) == 0:
        return (0, 0, mask.size[0], mask.size[1])
    
    min_y = max(0, white_pixels[0].min() - padding)
    max_y = min(mask.size[1], white_pixels[0].max() + padding)
    min_x = max(0, white_pixels[1].min() - padding)
    max_x = min(mask.size[0], white_pixels[1].max() + padding)
    
    return (min_x, min_y, max_x, max_y)


# =====================================================
# SEGMENTATION AUTOMATIQUE
# =====================================================

def instance_segment(
    image: Image.Image,
    semantic_mask: Image.Image = None,
    points: list[tuple[int, int]] = None,
    box: tuple[int, int, int, int] = None,
    save_path: str = None
) -> Image.Image:
    """
    Point d'entrée principal pour la segmentation par instance
    
    Choisit automatiquement la meilleure méthode:
    - Si semantic_mask fourni → affinage SAM2
    - Si points fournis → segmentation par points
    - Si box fournie → segmentation par box
    
    Args:
        image: Image PIL d'entrée
        semantic_mask: Masque sémantique optionnel
        points: Points optionnels
        box: Box optionnelle
        save_path: Chemin pour sauvegarder
    
    Returns:
        Masque segmenté
    """
    
    # Priorité 1: Affinage depuis masque sémantique
    if semantic_mask is not None:
        return instance_segment_from_semantic(
            image, semantic_mask, num_points=5, save_path=save_path
        )
    
    # Priorité 2: Points fournis
    if points is not None and len(points) > 0:
        return instance_segment_with_points(image, points, save_path=save_path)
    
    # Priorité 3: Box fournie
    if box is not None:
        return instance_segment_with_box(image, box, save_path=save_path)
    
    # Fallback: Masque vide
    print(f"   ⚠️  Aucune information de guidage fournie")
    return Image.new("L", image.size, 0)


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def get_mask_iou(mask1: Image.Image, mask2: Image.Image) -> float:
    """Calcule l'IoU entre deux masques"""
    
    m1 = np.array(mask1) > 127
    m2 = np.array(mask2) > 127
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_mask_coverage(mask: Image.Image) -> float:
    """Calcule le pourcentage de couverture du masque"""
    
    mask_array = np.array(mask)
    return np.sum(mask_array > 127) / mask_array.size
