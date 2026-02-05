# =====================================================
# ÉTAPE 6: MASK REFINEMENT
# =====================================================
# Nettoie et affine le masque final
# Opérations morphologiques, dilatation, feathering

import numpy as np
from PIL import Image, ImageFilter
from typing import Optional


def refine_mask(
    mask: Image.Image,
    dilate: int = 2,
    feather: int = 6,
    clean: bool = True,
    min_area: int = 100,
    fill_holes: bool = True,
    smooth_contours: bool = True
) -> Image.Image:
    """
    Affine un masque avec toutes les opérations de nettoyage
    
    Args:
        mask: Masque PIL à affiner
        dilate: Pixels de dilatation (0 = pas de dilatation)
        feather: Rayon de feathering (0 = pas de feathering)
        clean: Nettoyer les petites régions isolées
        min_area: Aire minimale des régions à garder
        fill_holes: Remplir les trous dans le masque
        smooth_contours: Lisser les contours
    
    Returns:
        Masque affiné
    """
    
    print(f"   ✨ Raffinement du masque...")
    
    result = mask.copy()
    
    # 1. Nettoyage morphologique
    if clean:
        result = clean_mask_morphology(result, min_area=min_area)
        print(f"      ✓ Nettoyage morphologique")
    
    # 2. Remplir les trous
    if fill_holes:
        result = fill_mask_holes(result)
        print(f"      ✓ Trous remplis")
    
    # 3. Lisser les contours
    if smooth_contours:
        result = smooth_mask_contours(result, iterations=2)
        print(f"      ✓ Contours lissés")
    
    # 4. Dilatation
    if dilate > 0:
        result = dilate_mask(result, iterations=dilate)
        print(f"      ✓ Dilatation: {dilate}px")
    
    # 5. Feathering (doit être fait en dernier)
    if feather > 0:
        result = feather_mask(result, radius=feather)
        print(f"      ✓ Feathering: {feather}px")
    
    print(f"   ✅ Masque raffiné")
    
    return result


# =====================================================
# OPÉRATIONS MORPHOLOGIQUES
# =====================================================

def clean_mask_morphology(
    mask: Image.Image,
    min_area: int = 100,
    open_iterations: int = 2,
    close_iterations: int = 2
) -> Image.Image:
    """
    Nettoie le masque avec des opérations morphologiques
    
    - Opening: Supprime les petits objets blancs (bruit)
    - Closing: Ferme les petits trous noirs
    """
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    
    # Opening (erosion + dilation): supprime le bruit
    if open_iterations > 0:
        mask_array = ndimage.binary_opening(mask_array, iterations=open_iterations)
    
    # Closing (dilation + erosion): ferme les trous
    if close_iterations > 0:
        mask_array = ndimage.binary_closing(mask_array, iterations=close_iterations)
    
    # Supprimer les petites régions
    if min_area > 0:
        labeled, num_features = ndimage.label(mask_array)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_area:
                mask_array[region] = False
    
    return Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")


def fill_mask_holes(mask: Image.Image) -> Image.Image:
    """Remplit les trous dans le masque"""
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    
    # Remplir les trous
    filled = ndimage.binary_fill_holes(mask_array)
    
    return Image.fromarray((filled * 255).astype(np.uint8), mode="L")


def smooth_mask_contours(
    mask: Image.Image,
    iterations: int = 2
) -> Image.Image:
    """Lisse les contours du masque"""
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    
    for _ in range(iterations):
        # Erosion puis dilation = lissage
        mask_array = ndimage.binary_erosion(mask_array)
        mask_array = ndimage.binary_dilation(mask_array)
    
    return Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")


# =====================================================
# DILATATION
# =====================================================

def dilate_mask(
    mask: Image.Image,
    iterations: int = 3,
    kernel_size: int = 3
) -> Image.Image:
    """
    Dilate le masque (agrandit la zone blanche)
    
    Args:
        mask: Masque à dilater
        iterations: Nombre d'itérations
        kernel_size: Taille du kernel (3, 5, 7)
    
    Returns:
        Masque dilaté
    """
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    
    # Créer le kernel
    if kernel_size == 3:
        kernel = np.ones((3, 3), dtype=bool)
    elif kernel_size == 5:
        kernel = np.ones((5, 5), dtype=bool)
    else:
        kernel = np.ones((7, 7), dtype=bool)
    
    # Dilater
    dilated = ndimage.binary_dilation(
        mask_array,
        structure=kernel,
        iterations=iterations
    )
    
    return Image.fromarray((dilated * 255).astype(np.uint8), mode="L")


def erode_mask(
    mask: Image.Image,
    iterations: int = 3,
    kernel_size: int = 3
) -> Image.Image:
    """
    Érode le masque (rétrécit la zone blanche)
    """
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    
    if kernel_size == 3:
        kernel = np.ones((3, 3), dtype=bool)
    elif kernel_size == 5:
        kernel = np.ones((5, 5), dtype=bool)
    else:
        kernel = np.ones((7, 7), dtype=bool)
    
    eroded = ndimage.binary_erosion(
        mask_array,
        structure=kernel,
        iterations=iterations
    )
    
    return Image.fromarray((eroded * 255).astype(np.uint8), mode="L")


# =====================================================
# FEATHERING
# =====================================================

def feather_mask(
    mask: Image.Image,
    radius: int = 8
) -> Image.Image:
    """
    Applique un feathering (adoucissement des bords)
    
    Args:
        mask: Masque à adoucir
        radius: Rayon du blur gaussien
    
    Returns:
        Masque avec bords adoucis (gradients)
    """
    
    # Appliquer un blur gaussien
    feathered = mask.filter(ImageFilter.GaussianBlur(radius))
    
    return feathered


def feather_mask_adaptive(
    mask: Image.Image,
    image_size: tuple
) -> Image.Image:
    """
    Feathering adaptatif selon la taille de l'image
    
    Petite image → feather faible
    Grande image → feather fort
    """
    
    # Calculer le rayon adaptatif
    max_dim = max(image_size)
    
    if max_dim < 512:
        radius = 2
    elif max_dim < 1024:
        radius = 4
    elif max_dim < 2048:
        radius = 8
    else:
        radius = 12
    
    return feather_mask(mask, radius)


def feather_mask_by_coverage(
    mask: Image.Image,
    min_radius: int = 2,
    max_radius: int = 12
) -> Image.Image:
    """
    Feathering selon la couverture du masque
    
    Petite zone → feather faible
    Grande zone → feather fort
    """
    
    mask_array = np.array(mask)
    coverage = np.sum(mask_array > 127) / mask_array.size
    
    # Interpoler le rayon
    radius = int(min_radius + (max_radius - min_radius) * coverage)
    
    return feather_mask(mask, radius)


# =====================================================
# PARAMÈTRES DYNAMIQUES
# =====================================================

def get_dynamic_refinement_params(
    mask: Image.Image,
    image_size: tuple
) -> dict:
    """
    Calcule les paramètres de raffinement optimaux
    
    Args:
        mask: Masque à raffiner
        image_size: Taille de l'image originale
    
    Returns:
        Dict avec dilate, feather, clean, etc.
    """
    
    mask_array = np.array(mask)
    coverage = np.sum(mask_array > 127) / mask_array.size
    max_dim = max(image_size)
    
    params = {
        "dilate": 0,
        "feather": 0,
        "clean": True,
        "min_area": 100,
        "fill_holes": True,
        "smooth_contours": True
    }
    
    # Ajuster selon la taille de l'image
    if max_dim < 512:
        params["dilate"] = 1
        params["feather"] = 2
        params["min_area"] = 50
    elif max_dim < 1024:
        params["dilate"] = 2
        params["feather"] = 4
        params["min_area"] = 100
    elif max_dim < 2048:
        params["dilate"] = 3
        params["feather"] = 6
        params["min_area"] = 200
    else:
        params["dilate"] = 4
        params["feather"] = 8
        params["min_area"] = 400
    
    # Ajuster selon la couverture
    if coverage < 0.1:
        # Petite zone: feather faible pour garder la précision
        params["feather"] = max(2, params["feather"] // 2)
    elif coverage > 0.5:
        # Grande zone: feather plus fort pour transitions douces
        params["feather"] = min(12, params["feather"] * 2)
    
    return params


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def invert_mask(mask: Image.Image) -> Image.Image:
    """Inverse un masque (blanc ↔ noir)"""
    
    mask_array = np.array(mask)
    inverted = 255 - mask_array
    
    return Image.fromarray(inverted, mode="L")


def threshold_mask(mask: Image.Image, threshold: int = 127) -> Image.Image:
    """Binarise un masque selon un seuil"""
    
    mask_array = np.array(mask)
    binary = (mask_array > threshold).astype(np.uint8) * 255
    
    return Image.fromarray(binary, mode="L")


def get_mask_edge(mask: Image.Image, width: int = 3) -> Image.Image:
    """Extrait les bords du masque"""
    
    dilated = dilate_mask(mask, iterations=width)
    eroded = erode_mask(mask, iterations=width)
    
    dilated_array = np.array(dilated)
    eroded_array = np.array(eroded)
    
    edge = dilated_array - eroded_array
    
    return Image.fromarray(edge.astype(np.uint8), mode="L")
