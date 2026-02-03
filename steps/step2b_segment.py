# Step 2b: Segmentation avec SAM2 / SegFormer
import numpy as np
from PIL import Image


def segment_target_region(
    image: Image.Image,
    target: str = "floor",
    method: str = "auto",
    points: list[tuple[int, int]] = None,
    box: tuple[int, int, int, int] = None,
    dilate: int = 3,
    feather: int = 8,
    save_path: str = None
) -> Image.Image:
    """
    Segmente une région cible de l'image
    
    Args:
        image: Image PIL d'entrée
        target: Cible à segmenter ("floor", "wall", "ceiling", "custom")
        method: Méthode de segmentation ("auto", "points", "box")
        points: Points pour la méthode "points" [(x,y), ...]
        box: Bounding box pour la méthode "box" (x1, y1, x2, y2)
        dilate: Nombre d'itérations de dilatation du masque
        feather: Rayon de feathering pour adoucir les bords
        save_path: Chemin pour sauvegarder le masque
    
    Returns:
        Masque PIL (blanc = zone à modifier, noir = zone à préserver)
    """
    from models.sam2 import (
        segment_floor_auto,
        segment_with_points_sam2,
        segment_with_box_sam2,
        dilate_mask,
        feather_mask
    )
    
    print(f"   🎯 Segmentation: target={target}, method={method}")
    
    # Segmentation selon la méthode
    if method == "auto":
        if target == "floor":
            mask = segment_floor_auto(image)
        else:
            # Pour d'autres targets, utiliser des heuristiques
            mask = segment_floor_auto(image)  # Fallback
            
    elif method == "points":
        if points is None:
            # Points par défaut au centre-bas pour le sol
            w, h = image.size
            points = [
                (w // 2, int(h * 0.8)),      # Centre bas
                (w // 4, int(h * 0.85)),     # Gauche bas
                (3 * w // 4, int(h * 0.85)), # Droite bas
            ]
        mask = segment_with_points_sam2(image, points)
        
    elif method == "box":
        if box is None:
            # Box par défaut pour le tiers inférieur (sol)
            w, h = image.size
            box = (0, int(h * 0.6), w, h)
        mask = segment_with_box_sam2(image, box)
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    # Post-traitement du masque
    if dilate > 0:
        print(f"   🔄 Dilatation du masque ({dilate} itérations)")
        mask = dilate_mask(mask, iterations=dilate)
    
    if feather > 0:
        print(f"   🌫️  Feathering du masque (rayon={feather})")
        mask = feather_mask(mask, radius=feather)
    
    # Sauvegarder si demandé
    if save_path:
        mask.save(save_path)
        print(f"   💾 Masque final sauvegardé: {save_path}")
    
    # Stats
    mask_np = np.array(mask)
    coverage = np.sum(mask_np > 128) / mask_np.size * 100
    print(f"   ✅ Masque généré: {coverage:.1f}% de couverture")
    
    return mask


def create_masked_image(
    image: Image.Image,
    mask: Image.Image,
    save_path: str = None
) -> Image.Image:
    """
    Crée une image avec la zone masquée visible (pour debug)
    """
    import numpy as np
    
    img_np = np.array(image)
    mask_np = np.array(mask.convert("L"))
    
    # Créer une overlay rouge semi-transparente
    overlay = img_np.copy()
    overlay[mask_np > 128] = [255, 0, 0]  # Rouge pour les zones masquées
    
    # Mélanger avec l'original
    result = (0.5 * img_np + 0.5 * overlay).astype(np.uint8)
    result_image = Image.fromarray(result)
    
    if save_path:
        result_image.save(save_path)
        print(f"   💾 Preview masque sauvegardé: {save_path}")
    
    return result_image


def invert_mask(mask: Image.Image) -> Image.Image:
    """
    Inverse le masque (blanc <-> noir)
    """
    import numpy as np
    
    mask_np = np.array(mask)
    inverted = 255 - mask_np
    
    return Image.fromarray(inverted, mode="L")
