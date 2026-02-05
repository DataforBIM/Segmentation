# =====================================================
# SPATIAL ZONES DETECTOR
# =====================================================
# Détecte les zones spatiales pour les actions "ADD"
# Zone d'accueil plutôt que l'objet final

import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SpatialZone:
    """Représente une zone spatiale détectée"""
    name: str
    mask: Image.Image
    region: str  # "foreground", "midground", "background"
    position: str  # "top", "middle", "bottom", "left", "right", "center"
    area_ratio: float  # Ratio de la zone par rapport à l'image totale
    confidence: float


# =====================================================
# DÉTECTION DE ZONES SPATIALES
# =====================================================

def detect_spatial_zone(
    image: Image.Image,
    zone_description: str,
    semantic_masks: Optional[Dict[str, Image.Image]] = None,
    depth_map: Optional[np.ndarray] = None
) -> SpatialZone:
    """
    Détecte une zone spatiale depuis description naturelle
    
    Args:
        image: Image PIL
        zone_description: Description de la zone ("garden_foreground", "lawn_edge", etc.)
        semantic_masks: Masques sémantiques disponibles
        depth_map: Carte de profondeur (optionnel)
    
    Returns:
        SpatialZone avec masque de la zone
    
    Examples:
        >>> zone = detect_spatial_zone(image, "garden_foreground")
        >>> zone = detect_spatial_zone(image, "flowerbed_zone")
    """
    
    # Parser la description de zone
    zone_parts = zone_description.lower().split("_")
    
    # Extraire contexte et position
    context = None
    position = None
    
    for part in zone_parts:
        if part in ["foreground", "midground", "background", "front", "middle", "back"]:
            position = _normalize_position(part)
        elif part in ["garden", "lawn", "ground", "floor", "wall", "vegetation"]:
            context = part
    
    # Si pas de contexte, détecter depuis l'image
    if context is None:
        context = "ground"  # Défaut
    
    # Si pas de position, défaut = foreground
    if position is None:
        position = "foreground"
    
    # Créer le masque de zone
    zone_mask = create_zone_mask(
        image=image,
        context=context,
        position=position,
        semantic_masks=semantic_masks,
        depth_map=depth_map
    )
    
    # Calculer métriques
    w, h = image.size
    total_pixels = w * h
    zone_pixels = np.sum(np.array(zone_mask) > 127)
    area_ratio = zone_pixels / total_pixels
    
    # Déterminer région
    region = _determine_region(position)
    
    # Déterminer position générale
    pos_general = _determine_position(zone_mask, image.size)
    
    return SpatialZone(
        name=zone_description,
        mask=zone_mask,
        region=region,
        position=pos_general,
        area_ratio=area_ratio,
        confidence=0.8
    )


def create_zone_mask(
    image: Image.Image,
    context: str,
    position: str,
    semantic_masks: Optional[Dict[str, Image.Image]] = None,
    depth_map: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Crée un masque de zone basé sur contexte + position
    
    Args:
        image: Image PIL
        context: Contexte ("garden", "lawn", "ground", "floor", etc.)
        position: Position ("foreground", "midground", "background")
        semantic_masks: Masques sémantiques
        depth_map: Carte de profondeur
    
    Returns:
        Masque PIL de la zone
    """
    
    w, h = image.size
    
    # Étape 1: Récupérer le masque de contexte
    context_mask = _get_context_mask(context, semantic_masks, (w, h))
    context_array = np.array(context_mask)
    
    # Étape 2: Appliquer filtre de position
    if depth_map is not None:
        # Utiliser depth map pour foreground/background
        position_mask = _get_position_mask_from_depth(depth_map, position, (w, h))
    else:
        # Fallback: utiliser position géométrique
        position_mask = _get_position_mask_geometric(position, (w, h))
    
    position_array = np.array(position_mask)
    
    # Étape 3: Intersection context ∩ position
    zone_array = np.minimum(context_array, position_array)
    
    # Étape 4: Soustraire objets existants (arbres, structures)
    if semantic_masks:
        zone_array = _subtract_existing_objects(zone_array, semantic_masks)
    
    return Image.fromarray(zone_array.astype(np.uint8))


def _get_context_mask(
    context: str,
    semantic_masks: Optional[Dict[str, Image.Image]],
    size: Tuple[int, int]
) -> Image.Image:
    """Récupère le masque de contexte depuis semantic masks"""
    
    if semantic_masks is None:
        # Pas de masques sémantiques → retourner tout l'image
        return Image.new("L", size, 255)
    
    # Mapper contexte → classes sémantiques
    context_mapping = {
        "garden": ["grass", "ground", "vegetation"],
        "lawn": ["grass", "ground"],
        "ground": ["grass", "ground", "ground"],
        "floor": ["floor"],
        "wall": ["wall", "building"],
        "vegetation": ["vegetation", "tree", "plant"],
        "sky": ["sky"]
    }
    
    classes = context_mapping.get(context, ["ground"])
    
    # Fusionner tous les masques correspondants
    mask_array = np.zeros(size[::-1], dtype=np.uint8)  # (h, w)
    
    for class_name in classes:
        if class_name in semantic_masks:
            class_mask = np.array(semantic_masks[class_name])
            mask_array = np.maximum(mask_array, class_mask)
    
    if np.sum(mask_array) == 0:
        # Aucun masque trouvé → retourner bas de l'image par défaut
        w, h = size
        mask_array = np.zeros((h, w), dtype=np.uint8)
        mask_array[int(h * 0.5):, :] = 255  # Moitié basse
    
    return Image.fromarray(mask_array)


def _get_position_mask_from_depth(
    depth_map: np.ndarray,
    position: str,
    size: Tuple[int, int]
) -> Image.Image:
    """Crée un masque de position depuis depth map"""
    
    # Normaliser depth map entre 0-255
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    
    # Foreground = faible profondeur (proche)
    # Background = grande profondeur (loin)
    
    if position == "foreground":
        # Garder les 33% les plus proches
        threshold = np.percentile(depth_map, 33)
        mask = (depth_map <= threshold).astype(np.uint8) * 255
    elif position == "background":
        # Garder les 33% les plus loin
        threshold = np.percentile(depth_map, 67)
        mask = (depth_map >= threshold).astype(np.uint8) * 255
    else:  # midground
        # Garder le milieu
        low_threshold = np.percentile(depth_map, 33)
        high_threshold = np.percentile(depth_map, 67)
        mask = ((depth_map > low_threshold) & (depth_map < high_threshold)).astype(np.uint8) * 255
    
    return Image.fromarray(mask)


def _get_position_mask_geometric(
    position: str,
    size: Tuple[int, int]
) -> Image.Image:
    """Crée un masque de position géométrique (sans depth map)"""
    
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if position == "foreground":
        # Bas de l'image (zone proche)
        mask[int(h * 0.6):, :] = 255
    elif position == "background":
        # Haut de l'image (zone loin)
        mask[:int(h * 0.4), :] = 255
    else:  # midground
        # Milieu
        mask[int(h * 0.3):int(h * 0.7), :] = 255
    
    # Ajouter un gradient progressif pour plus de naturel
    mask = _apply_gradient(mask, position)
    
    return Image.fromarray(mask)


def _apply_gradient(mask: np.ndarray, position: str) -> np.ndarray:
    """Applique un gradient doux aux bordures du masque"""
    
    from scipy.ndimage import gaussian_filter
    
    # Flouter légèrement les bords
    mask_float = mask.astype(np.float32)
    mask_smoothed = gaussian_filter(mask_float, sigma=20)
    
    return mask_smoothed.astype(np.uint8)


def _subtract_existing_objects(
    zone_array: np.ndarray,
    semantic_masks: Dict[str, Image.Image]
) -> np.ndarray:
    """Soustrait les objets existants de la zone (arbres, structures)"""
    
    # Objets à soustraire (ne pas générer par-dessus)
    objects_to_subtract = ["tree", "building", "person", "car", "furniture"]
    
    for obj_name in objects_to_subtract:
        if obj_name in semantic_masks:
            obj_mask = np.array(semantic_masks[obj_name])
            # Soustraire: zone = zone - objet
            zone_array = np.where(obj_mask > 127, 0, zone_array)
    
    return zone_array


def _normalize_position(position: str) -> str:
    """Normalise les variants de position"""
    
    mapping = {
        "front": "foreground",
        "back": "background",
        "middle": "midground",
        "close": "foreground",
        "far": "background"
    }
    
    return mapping.get(position, position)


def _determine_region(position: str) -> str:
    """Détermine la région depuis la position"""
    
    mapping = {
        "foreground": "foreground",
        "midground": "midground",
        "background": "background"
    }
    
    return mapping.get(position, "foreground")


def _determine_position(mask: Image.Image, size: Tuple[int, int]) -> str:
    """Détermine la position générale du masque (top/middle/bottom/center)"""
    
    mask_array = np.array(mask)
    w, h = size
    
    # Trouver le centre de masse du masque
    y_coords, x_coords = np.where(mask_array > 127)
    
    if len(y_coords) == 0:
        return "center"
    
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    # Déterminer position verticale
    if center_y < h * 0.33:
        v_pos = "top"
    elif center_y > h * 0.67:
        v_pos = "bottom"
    else:
        v_pos = "middle"
    
    # Déterminer position horizontale
    if center_x < w * 0.33:
        h_pos = "left"
    elif center_x > w * 0.67:
        h_pos = "right"
    else:
        h_pos = "center"
    
    # Combiner
    if h_pos == "center":
        return v_pos
    else:
        return f"{v_pos}_{h_pos}"


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def describe_zone(zone: SpatialZone) -> str:
    """Description lisible de la zone"""
    
    return (
        f"{zone.name} | "
        f"Region: {zone.region} | "
        f"Position: {zone.position} | "
        f"Area: {zone.area_ratio:.1%} | "
        f"Confidence: {zone.confidence:.0%}"
    )


def visualize_zone(
    image: Image.Image,
    zone: SpatialZone,
    alpha: float = 0.5
) -> Image.Image:
    """Visualise la zone sur l'image"""
    
    from PIL import ImageDraw
    
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    
    # Créer overlay semi-transparent
    zone_array = np.array(zone.mask)
    overlay_array = np.array(overlay)
    
    # Zone en vert semi-transparent
    green_overlay = np.zeros_like(overlay_array)
    green_overlay[..., 1] = 255  # Canal vert
    
    # Appliquer avec alpha
    mask_3d = np.stack([zone_array] * 3, axis=-1) / 255.0
    result = overlay_array * (1 - mask_3d * alpha) + green_overlay * mask_3d * alpha
    
    return Image.fromarray(result.astype(np.uint8))
