# =====================================================
# ÉTAPE 5: MASK FUSION + PRIORITIES
# =====================================================
# Fusionne les masques avec un système de priorités
# Crée la structure hiérarchique: target / protected / context

import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional

from .semantic_segmentation import SemanticMap, get_combined_mask
from .target_resolver import Target


@dataclass
class MaskLayers:
    """Structure des masques hiérarchiques (comme ChatGPT)"""
    
    # Masque de la zone à modifier
    target: Optional[Image.Image] = None
    
    # Masque des zones protégées (ne jamais toucher)
    protected: Optional[Image.Image] = None
    
    # Masque de contexte (aide à la cohérence)
    context: Optional[Image.Image] = None
    
    # Masque final après fusion
    final: Optional[Image.Image] = None
    
    # Métadonnées
    coverage: float = 0.0
    confidence: float = 1.0
    method_used: str = "semantic"


# =====================================================
# FUSION PRINCIPALE
# =====================================================

def fuse_masks(
    semantic_map: SemanticMap,
    target: Target,
    instance_mask: Image.Image = None,
    save_path: str = None
) -> MaskLayers:
    """
    Fusionne les masques selon les priorités définies
    
    Args:
        semantic_map: Map sémantique de l'image
        target: Cibles résolues (primary, protected, context)
        instance_mask: Masque SAM2 optionnel pour affinement
        save_path: Chemin pour sauvegarder
    
    Returns:
        MaskLayers avec target, protected, context, final
    
    Example:
        >>> layers = fuse_masks(semantic_map, target)
        >>> final_mask = layers.final  # Masque prêt pour SDXL
    """
    
    print("   🔀 Fusion des masques avec priorités...")
    
    # 1. CRÉER LE MASQUE TARGET (zone à modifier)
    target_mask = _create_target_mask(semantic_map, target, instance_mask)
    
    # 2. CRÉER LE MASQUE PROTECTED (zones à préserver)
    protected_mask = _create_protected_mask(semantic_map, target)
    
    # 3. CRÉER LE MASQUE CONTEXT (pour cohérence)
    context_mask = _create_context_mask(semantic_map, target)
    
    # 4. FUSION FINALE: target - protected
    final_mask = _apply_protection(target_mask, protected_mask)
    
    # 5. Calculer la couverture
    coverage = _calculate_coverage(final_mask)
    
    print(f"   ✅ Masques fusionnés (couverture: {coverage*100:.1f}%)")
    
    # Sauvegarder si demandé
    if save_path:
        _save_mask_layers(
            target_mask, protected_mask, context_mask, final_mask,
            save_path
        )
    
    return MaskLayers(
        target=target_mask,
        protected=protected_mask,
        context=context_mask,
        final=final_mask,
        coverage=coverage,
        confidence=target.confidence,
        method_used=target.method
    )


# =====================================================
# CRÉATION DES MASQUES
# =====================================================

def _create_target_mask(
    semantic_map: SemanticMap,
    target: Target,
    instance_mask: Image.Image = None
) -> Image.Image:
    """Crée le masque de la zone à modifier"""
    
    # Si un masque SAM2 est fourni, l'utiliser
    if instance_mask is not None:
        print(f"      → Target: utilisation du masque SAM2")
        return instance_mask
    
    # Sinon, combiner les masques sémantiques des cibles primaires
    if not target.primary:
        return Image.new("L", semantic_map.size, 0)
    
    masks_to_combine = []
    for class_name in target.primary:
        if class_name in semantic_map.masks:
            masks_to_combine.append(semantic_map.masks[class_name])
    
    if not masks_to_combine:
        print(f"      ⚠️  Aucun masque trouvé pour: {target.primary}")
        return Image.new("L", semantic_map.size, 0)
    
    # Fusionner (union)
    combined = np.zeros((semantic_map.size[1], semantic_map.size[0]), dtype=np.uint8)
    for mask in masks_to_combine:
        combined = np.maximum(combined, np.array(mask))
    
    print(f"      → Target: {', '.join(target.primary)}")
    
    return Image.fromarray(combined, mode="L")


def _create_protected_mask(
    semantic_map: SemanticMap,
    target: Target
) -> Image.Image:
    """Crée le masque des zones à protéger"""
    
    if not target.protected:
        return Image.new("L", semantic_map.size, 0)
    
    masks_to_combine = []
    protected_found = []
    
    for class_name in target.protected:
        if class_name in semantic_map.masks:
            masks_to_combine.append(semantic_map.masks[class_name])
            protected_found.append(class_name)
    
    if not masks_to_combine:
        return Image.new("L", semantic_map.size, 0)
    
    # Fusionner (union)
    combined = np.zeros((semantic_map.size[1], semantic_map.size[0]), dtype=np.uint8)
    for mask in masks_to_combine:
        combined = np.maximum(combined, np.array(mask))
    
    print(f"      → Protected: {', '.join(protected_found)}")
    
    return Image.fromarray(combined, mode="L")


def _create_context_mask(
    semantic_map: SemanticMap,
    target: Target
) -> Image.Image:
    """Crée le masque de contexte"""
    
    if not target.context:
        return Image.new("L", semantic_map.size, 0)
    
    masks_to_combine = []
    
    for class_name in target.context:
        if class_name in semantic_map.masks:
            masks_to_combine.append(semantic_map.masks[class_name])
    
    if not masks_to_combine:
        return Image.new("L", semantic_map.size, 0)
    
    # Fusionner (union)
    combined = np.zeros((semantic_map.size[1], semantic_map.size[0]), dtype=np.uint8)
    for mask in masks_to_combine:
        combined = np.maximum(combined, np.array(mask))
    
    return Image.fromarray(combined, mode="L")


def _apply_protection(
    target_mask: Image.Image,
    protected_mask: Image.Image
) -> Image.Image:
    """
    Applique la protection: final = target - protected
    
    Les zones protégées sont soustraites du masque cible
    """
    
    target_array = np.array(target_mask)
    protected_array = np.array(protected_mask)
    
    # Soustraire: où protected est blanc, final devient noir
    final_array = np.where(protected_array > 127, 0, target_array)
    
    return Image.fromarray(final_array.astype(np.uint8), mode="L")


def _calculate_coverage(mask: Image.Image) -> float:
    """Calcule le pourcentage de couverture du masque"""
    
    mask_array = np.array(mask)
    return np.sum(mask_array > 127) / mask_array.size


# =====================================================
# SAUVEGARDE ET VISUALISATION
# =====================================================

def _save_mask_layers(
    target: Image.Image,
    protected: Image.Image,
    context: Image.Image,
    final: Image.Image,
    base_path: str
):
    """Sauvegarde tous les masques"""
    
    import os
    
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    
    # Sauvegarder chaque couche
    target.save(os.path.join(base_dir, f"{base_name}_target.png"))
    protected.save(os.path.join(base_dir, f"{base_name}_protected.png"))
    context.save(os.path.join(base_dir, f"{base_name}_context.png"))
    final.save(os.path.join(base_dir, f"{base_name}_final.png"))
    
    # Créer une visualisation combinée
    _create_combined_visualization(target, protected, context, final, base_path)
    
    print(f"   💾 Masques sauvegardés: {base_path}")


def _create_combined_visualization(
    target: Image.Image,
    protected: Image.Image,
    context: Image.Image,
    final: Image.Image,
    save_path: str
):
    """Crée une visualisation colorée des couches"""
    
    size = target.size
    
    # Créer une image RGB
    result = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Rouge = target
    target_array = np.array(target)
    result[:, :, 0] = np.where(target_array > 127, 200, 0)
    
    # Vert = context
    context_array = np.array(context)
    result[:, :, 1] = np.where(context_array > 127, 150, result[:, :, 1])
    
    # Bleu = protected
    protected_array = np.array(protected)
    result[:, :, 2] = np.where(protected_array > 127, 200, 0)
    
    # Sauvegarder
    Image.fromarray(result).save(save_path.replace(".png", "_visualization.png"))


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def expand_mask(mask: Image.Image, pixels: int = 5) -> Image.Image:
    """Étend un masque de quelques pixels"""
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    expanded = ndimage.binary_dilation(mask_array, iterations=pixels)
    
    return Image.fromarray((expanded * 255).astype(np.uint8), mode="L")


def shrink_mask(mask: Image.Image, pixels: int = 5) -> Image.Image:
    """Rétrécit un masque de quelques pixels"""
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    shrunk = ndimage.binary_erosion(mask_array, iterations=pixels)
    
    return Image.fromarray((shrunk * 255).astype(np.uint8), mode="L")


def smooth_mask_edges(mask: Image.Image, radius: int = 3) -> Image.Image:
    """Lisse les bords d'un masque"""
    
    from PIL import ImageFilter
    
    smoothed = mask.filter(ImageFilter.GaussianBlur(radius))
    
    # Re-binariser
    smoothed_array = np.array(smoothed)
    binary = (smoothed_array > 127).astype(np.uint8) * 255
    
    return Image.fromarray(binary, mode="L")


def combine_masks_weighted(
    masks: list[Image.Image],
    weights: list[float]
) -> Image.Image:
    """Combine plusieurs masques avec des poids"""
    
    if not masks:
        return None
    
    size = masks[0].size
    combined = np.zeros((size[1], size[0]), dtype=np.float32)
    
    for mask, weight in zip(masks, weights):
        mask_array = np.array(mask).astype(np.float32) / 255.0
        combined += mask_array * weight
    
    # Normaliser et binariser
    combined = np.clip(combined, 0, 1)
    binary = (combined > 0.5).astype(np.uint8) * 255
    
    return Image.fromarray(binary, mode="L")
