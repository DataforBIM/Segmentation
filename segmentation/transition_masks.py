# =====================================================
# TRANSITION MASKS - Blending progressif
# =====================================================
# Crée des masques de transition pour intégration douce

import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Dict
from dataclasses import dataclass

# Import pour feathering adaptatif
try:
    from .mask_refinement import get_dynamic_refinement_params, feather_mask
    MASK_REFINEMENT_AVAILABLE = True
except ImportError:
    MASK_REFINEMENT_AVAILABLE = False


@dataclass
class TransitionMasks:
    """Ensemble de masques pour blending progressif"""
    
    # Masque principal (100% modification)
    core: Image.Image
    
    # Masque de transition (gradient 100% → 0%)
    transition: Image.Image
    
    # Masque combiné (core + transition)
    combined: Image.Image
    
    # Largeur de transition en pixels
    transition_width: int
    
    # Type de gradient ("linear", "gaussian", "cosine")
    gradient_type: str


# =====================================================
# GÉNÉRATION DE MASQUES DE TRANSITION
# =====================================================

def create_transition_masks(
    mask_core: Image.Image,
    transition_width: int = 12,
    gradient_type: str = "cosine",
    feather_strength: float = 0.5,
    adaptive_feather: bool = True
) -> TransitionMasks:
    """
    Crée des masques de transition pour blending progressif
    
    Architecture:
    1. mask_core: Zone principale (100% modification) [Objet]
    2. mask_transition: Bord élargi avec gradient (100% → 0%) ✨ FEATHERING ADAPTATIF [Transition]
    3. mask_combined: Union des deux (pour inpainting)
    
    Args:
        mask_core: Masque principal PIL
        transition_width: Largeur de la zone de transition en pixels
        gradient_type: Type de gradient ("linear", "gaussian", "cosine")
        feather_strength: Force du feathering (0.0-1.0) - utilisé si adaptive_feather=False
        adaptive_feather: Si True, calcule le feather radius adaptatif basé sur l'aire du masque
    
    Returns:
        TransitionMasks avec core, transition, et combined
    
    Example:
        >>> masks = create_transition_masks(mask, transition_width=12, adaptive_feather=True)
        >>> # Utiliser masks.combined pour inpainting
        >>> # Utiliser masks.core + masks.transition pour blending
    """
    
    # Convertir en array
    core_array = np.array(mask_core).astype(np.float32) / 255.0
    
    # 1. DILATATION: Élargir le masque
    dilated = _dilate_mask(core_array, transition_width)
    
    # 2. TRANSITION: Zone entre dilaté et core
    transition_array = dilated - core_array
    transition_array = np.clip(transition_array, 0, 1)
    
    # 3. GRADIENT: Appliquer gradient progressif sur transition
    transition_with_gradient = _apply_gradient(
        transition_array,
        core_array,
        gradient_type=gradient_type,
        feather_strength=feather_strength if not adaptive_feather else 0.0
    )
    
    # Convertir en PIL
    core_pil = mask_core.copy()
    transition_pil = Image.fromarray((transition_with_gradient * 255).astype(np.uint8))
    
    # ✨ 3.5: FEATHERING ADAPTATIF sur mask_transition
    if adaptive_feather and MASK_REFINEMENT_AVAILABLE:
        # Calculer le feather radius adaptatif basé sur l'aire du masque
        params = get_dynamic_refinement_params(transition_pil, transition_pil.size)
        feather_radius = params["feather"]
        
        # Appliquer feathering adaptatif
        transition_pil = feather_mask(transition_pil, radius=feather_radius)
        transition_with_gradient = np.array(transition_pil).astype(np.float32) / 255.0
    
    # 4. COMBINED: Union core (100%) + transition (gradient)
    combined_array = np.clip(core_array + transition_with_gradient, 0, 1)
    combined_pil = Image.fromarray((combined_array * 255).astype(np.uint8))
    
    return TransitionMasks(
        core=core_pil,
        transition=transition_pil,
        combined=combined_pil,
        transition_width=transition_width,
        gradient_type=gradient_type
    )


def _dilate_mask(mask_array: np.ndarray, width: int) -> np.ndarray:
    """Dilate le masque de width pixels (sans scipy)"""
    
    if width <= 0:
        return mask_array
    
    # Convertir en PIL pour utiliser les filtres
    mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
    
    # Dilater en appliquant MaxFilter plusieurs fois
    for _ in range(width):
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
    
    # Reconvertir
    dilated = np.array(mask_pil).astype(np.float32) / 255.0
    
    return dilated


def _apply_gradient(
    transition_array: np.ndarray,
    core_array: np.ndarray,
    gradient_type: str = "cosine",
    feather_strength: float = 0.5
) -> np.ndarray:
    """
    Applique un gradient progressif sur la zone de transition (sans scipy)
    
    Args:
        transition_array: Zone de transition binaire (0 ou 1)
        core_array: Zone core (référence pour distance)
        gradient_type: "linear", "gaussian", "cosine"
        feather_strength: Force du feathering
    
    Returns:
        Transition avec gradient (1.0 près du core → 0.0 vers l'extérieur)
    """
    
    # Calculer la distance depuis le core (approximation simple)
    distances = _compute_distance_transform(core_array)
    
    # Ne garder que les distances dans la zone de transition
    distances_in_transition = distances * (transition_array > 0.5)
    
    if np.max(distances_in_transition) == 0:
        return transition_array
    
    # Normaliser distances (0 au bord du core, 1 au bord extérieur)
    max_dist = np.max(distances_in_transition)
    normalized_dist = distances_in_transition / max_dist
    
    # Appliquer le gradient selon le type
    if gradient_type == "linear":
        # Gradient linéaire: 1 → 0
        gradient = 1.0 - normalized_dist
    
    elif gradient_type == "cosine":
        # Gradient cosinus (plus doux): cos(0) = 1 → cos(π/2) = 0
        gradient = np.cos(normalized_dist * np.pi / 2)
    
    elif gradient_type == "gaussian":
        # Gradient gaussien
        gradient = np.exp(-normalized_dist * 3)
    
    else:
        # Défaut: linéaire
        gradient = 1.0 - normalized_dist
    
    # Ne garder que dans la zone de transition
    gradient = gradient * (transition_array > 0.5)
    
    # Appliquer feathering avec Gaussian Blur PIL
    if feather_strength > 0:
        gradient_pil = Image.fromarray((gradient * 255).astype(np.uint8))
        blur_radius = int(feather_strength * 3)
        gradient_pil = gradient_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        gradient = np.array(gradient_pil).astype(np.float32) / 255.0
    
    return gradient


def _compute_distance_transform(core_array: np.ndarray) -> np.ndarray:
    """
    Calcule la distance transform (approximation rapide)
    
    Utilise une approximation chamfer distance pour performance
    """
    
    # Convertir en binaire
    core_binary = (core_array > 0.5).astype(np.float32)
    
    h, w = core_array.shape
    
    # Initialiser distances
    distances = np.where(core_binary, 0, 9999)
    
    # Forward pass (top-left to bottom-right)
    for y in range(1, h):
        for x in range(1, w-1):
            if not core_binary[y, x]:
                distances[y, x] = min(
                    distances[y, x],
                    distances[y-1, x] + 1,
                    distances[y, x-1] + 1,
                    distances[y-1, x-1] + 1.414,
                    distances[y-1, x+1] + 1.414
                )
    
    # Backward pass (bottom-right to top-left)
    for y in range(h-2, -1, -1):
        for x in range(w-2, 0, -1):
            if not core_binary[y, x]:
                distances[y, x] = min(
                    distances[y, x],
                    distances[y+1, x] + 1,
                    distances[y, x+1] + 1,
                    distances[y+1, x+1] + 1.414,
                    distances[y+1, x-1] + 1.414
                )
    
    return distances.astype(np.float32)


# =====================================================
# BLENDING AVEC MASQUES DE TRANSITION
# =====================================================

def blend_with_transition(
    original_image: Image.Image,
    generated_image: Image.Image,
    transition_masks: TransitionMasks
) -> Image.Image:
    """
    Blend deux images avec masques de transition
    
    Architecture:
    - Core: 100% generated
    - Transition: Blend progressif (100% → 0%)
    - Exterior: 100% original
    
    Args:
        original_image: Image originale
        generated_image: Image générée
        transition_masks: Masques de transition
    
    Returns:
        Image blendée avec transition douce
    """
    
    orig_array = np.array(original_image).astype(np.float32)
    gen_array = np.array(generated_image).astype(np.float32)
    
    core_array = np.array(transition_masks.core).astype(np.float32) / 255.0
    transition_array = np.array(transition_masks.transition).astype(np.float32) / 255.0
    
    # Étendre les masques à 3 canaux si nécessaire
    if len(orig_array.shape) == 3:
        core_3d = np.stack([core_array] * 3, axis=-1)
        transition_3d = np.stack([transition_array] * 3, axis=-1)
    else:
        core_3d = core_array
        transition_3d = transition_array
    
    # Blending:
    # result = generated * (core + transition) + original * (1 - core - transition)
    
    # Zone core: 100% generated
    result = gen_array * core_3d
    
    # Zone transition: blend progressif
    result += gen_array * transition_3d + orig_array * (1 - transition_3d)
    
    # Zone extérieure: 100% original
    exterior_mask = 1 - core_3d - transition_3d
    result += orig_array * exterior_mask
    
    # Normaliser
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


# =====================================================
# VISUALISATION
# =====================================================

def visualize_transition_masks(
    image: Image.Image,
    transition_masks: TransitionMasks,
    save_path: str = None
) -> Image.Image:
    """
    Visualise les masques de transition en overlay coloré
    
    Colors:
    - Rouge: Core (zone principale)
    - Jaune: Transition (gradient)
    - Transparent: Extérieur
    """
    
    img_array = np.array(image).astype(np.float32)
    core_array = np.array(transition_masks.core).astype(np.float32) / 255.0
    transition_array = np.array(transition_masks.transition).astype(np.float32) / 255.0
    
    # Créer overlays colorés
    overlay = img_array.copy()
    
    # Core en rouge (opacité 0.5)
    core_mask_3d = np.stack([core_array] * 3, axis=-1)
    red_overlay = np.array([255, 0, 0], dtype=np.float32)
    overlay = overlay * (1 - core_mask_3d * 0.5) + red_overlay * core_mask_3d * 0.5
    
    # Transition en jaune (opacité 0.3)
    transition_mask_3d = np.stack([transition_array] * 3, axis=-1)
    yellow_overlay = np.array([255, 255, 0], dtype=np.float32)
    overlay = overlay * (1 - transition_mask_3d * 0.3) + yellow_overlay * transition_mask_3d * 0.3
    
    result = Image.fromarray(overlay.astype(np.uint8))
    
    if save_path:
        result.save(save_path)
    
    return result


def create_mask_comparison(
    transition_masks: TransitionMasks,
    save_path: str = None
) -> Image.Image:
    """
    Crée une comparaison visuelle des 3 masques
    
    Layout: [Core | Transition | Combined]
    """
    
    width = transition_masks.core.width
    height = transition_masks.core.height
    
    # Créer canvas
    canvas = Image.new("L", (width * 3, height), 0)
    
    # Coller les masques
    canvas.paste(transition_masks.core, (0, 0))
    canvas.paste(transition_masks.transition, (width, 0))
    canvas.paste(transition_masks.combined, (width * 2, 0))
    
    if save_path:
        canvas.save(save_path)
    
    return canvas


# =====================================================
# CALCUL ADAPTATIF DE LARGEUR DE TRANSITION
# =====================================================

def compute_adaptive_transition_width(
    mask: Image.Image,
    image_size: Tuple[int, int],
    base_width: int = 12
) -> int:
    """
    Calcule la largeur de transition adaptative selon:
    - Taille de l'image
    - Taille de la zone masquée
    
    Règles:
    - Petites zones → transition étroite
    - Grandes zones → transition large
    - Haute résolution → transition plus large
    """
    
    w, h = image_size
    image_area = w * h
    
    # Aire du masque
    mask_array = np.array(mask)
    mask_area = np.sum(mask_array > 127)
    
    # Ratio masque/image
    ratio = mask_area / image_area
    
    # Adapter selon la taille
    if ratio < 0.1:  # Petite zone
        width = int(base_width * 0.7)
    elif ratio > 0.5:  # Grande zone
        width = int(base_width * 1.5)
    else:  # Zone moyenne
        width = base_width
    
    # Adapter selon résolution
    base_resolution = 1024 * 1024
    resolution_factor = np.sqrt(image_area / base_resolution)
    width = int(width * resolution_factor)
    
    # Limites
    width = max(6, min(width, 32))
    
    return width


# =====================================================
# INTÉGRATION AU PIPELINE
# =====================================================

def prepare_inpainting_with_transition(
    original_image: Image.Image,
    mask_core: Image.Image,
    transition_width: int = 12,
    gradient_type: str = "cosine"
) -> Dict:
    """
    Prépare les masques pour inpainting avec transition
    
    Returns:
        {
            "mask_for_inpainting": mask_combined (zone à générer),
            "transition_masks": TransitionMasks (pour blending final),
            "transition_width": int
        }
    """
    
    # Calculer largeur adaptative si nécessaire
    if transition_width == "auto":
        transition_width = compute_adaptive_transition_width(
            mask_core,
            original_image.size
        )
    
    # Créer masques de transition
    transition_masks = create_transition_masks(
        mask_core,
        transition_width=transition_width,
        gradient_type=gradient_type
    )
    
    return {
        "mask_for_inpainting": transition_masks.combined,
        "transition_masks": transition_masks,
        "transition_width": transition_width
    }
