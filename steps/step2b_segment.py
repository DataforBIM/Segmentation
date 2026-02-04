# Step 2b: Segmentation avec SAM2 / SegFormer
import numpy as np
from PIL import Image


def segment_target_region(
    image: Image.Image,
    target: str = "floor",
    method: str = "auto",
    scene_type: str = None,
    points: list[tuple[int, int]] = None,
    box: tuple[int, int, int, int] = None,
    dilate: int = 3,
    feather: int = 8,
    save_path: str = None
) -> Image.Image:
    """
    Segmente une région cible de l'image selon la scène détectée
    
    Args:
        image: Image PIL d'entrée
        target: Cible à segmenter ("floor", "wall", "ceiling", "ears", "eyes", etc.)
        method: Méthode de segmentation ("auto", "points", "box")
        scene_type: Type de scène (ANIMAL, INTERIOR, EXTERIOR, PORTRAIT, PRODUCT)
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
        segment_animal_part,
        segment_interior_element,
        segment_exterior_element,
        segment_portrait_element,
        segment_aerial_elements,
        dilate_mask,
        feather_mask,
        clean_mask_morphology,
        simplify_mask_contours
    )
    
    print(f"   🎯 Segmentation: target={target}, method={method}, scene={scene_type}")
    
    # === SEGMENTATION SPÉCIALE POUR SCÈNES AÉRIENNES ===
    if scene_type == "AERIAL":
        print(f"   🚁 Mode aérien: Segmentation multi-éléments avec SAM2")
        aerial_result = segment_aerial_elements(image, save_path=save_path)
        
        # Retourner le masque combiné de tous les éléments
        # SDXL va améliorer tous les éléments détectés séparément
        
        # Sauvegarder les métadonnées TOUJOURS (même si aucun masque)
        _save_aerial_metadata(aerial_result, save_path)
        
        if aerial_result["combined_mask"] is not None:
            mask = aerial_result["combined_mask"]
        else:
            # Fallback si aucun élément détecté
            print(f"   ⚠️  Aucun élément aérien détecté, utilisation de masque complet")
            mask = Image.new("L", image.size, 255)  # Masque blanc complet
    
    # === ROUTING SELON LA SCÈNE (non-aérienne) ===
    else:
        # Définir les targets par catégorie
        animal_parts = ["ears", "eyes", "fur", "tail", "paws", "nose", "body"]
        interior_elements = ["floor", "wall", "ceiling", "furniture", "window", "door"]
        exterior_elements = ["sky", "ground", "vegetation", "building", "road"]
        portrait_elements = ["face", "hair", "lips", "skin", "clothing"]
        
        # Segmentation selon la méthode
        if method == "auto":
            # Router selon la scène ET le target
            if scene_type == "ANIMAL" or target in animal_parts:
                mask = segment_animal_part(image, target)
                
            elif scene_type == "INTERIOR" or target in interior_elements:
                mask = segment_interior_element(image, target)
                
            elif scene_type == "EXTERIOR" or target in exterior_elements:
                mask = segment_exterior_element(image, target)
                
            elif scene_type == "PORTRAIT" or target in portrait_elements:
                mask = segment_portrait_element(image, target)
                
            else:
                # Fallback: essayer avec le target générique
                if target in animal_parts:
                    mask = segment_animal_part(image, target)
                elif target in interior_elements:
                    mask = segment_interior_element(image, target)
                else:
                    mask = segment_floor_auto(image)  # Fallback ultime
                
        elif method == "points":
            if points is None:
                points = _get_default_points(image, target, scene_type)
            mask = segment_with_points_sam2(image, points)
            
        elif method == "box":
            if box is None:
                box = _get_default_box(image, target, scene_type)
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


def _get_default_points(image: Image.Image, target: str, scene_type: str = None) -> list:
    """Retourne des points par défaut selon la cible et la scène"""
    w, h = image.size
    
    # === INTÉRIEUR ===
    if target == "floor":
        return [
            (w // 2, int(h * 0.8)),      # Centre bas
            (w // 4, int(h * 0.85)),     # Gauche bas
            (3 * w // 4, int(h * 0.85)), # Droite bas
        ]
    elif target == "wall":
        return [
            (w // 2, int(h * 0.4)),      # Centre mur
            (w // 4, int(h * 0.3)),      # Gauche mur
            (3 * w // 4, int(h * 0.3)),  # Droite mur
        ]
    elif target == "ceiling":
        return [
            (w // 2, int(h * 0.1)),      # Centre plafond
        ]
    elif target == "furniture":
        return [
            (w // 2, int(h * 0.5)),      # Centre
        ]
        
    # === ANIMAUX ===
    elif target == "ears":
        return [
            (int(w * 0.3), int(h * 0.15)),  # Oreille gauche
            (int(w * 0.7), int(h * 0.15)),  # Oreille droite
        ]
    elif target == "eyes":
        return [
            (int(w * 0.35), int(h * 0.35)), # Oeil gauche
            (int(w * 0.65), int(h * 0.35)), # Oeil droit
        ]
    elif target == "nose":
        return [
            (int(w * 0.5), int(h * 0.5)),   # Centre du museau
        ]
    elif target == "paws":
        return [
            (int(w * 0.3), int(h * 0.85)),  # Patte avant gauche
            (int(w * 0.7), int(h * 0.85)),  # Patte avant droite
        ]
    elif target == "tail":
        return [
            (int(w * 0.1), int(h * 0.6)),   # Queue (côté)
        ]
    elif target in ["fur", "body"]:
        return [
            (w // 2, h // 2),              # Centre de l'animal
        ]
        
    # === EXTÉRIEUR ===
    elif target == "sky":
        return [
            (w // 2, int(h * 0.15)),       # Haut centre
            (w // 4, int(h * 0.1)),        # Haut gauche
        ]
    elif target == "vegetation":
        return [
            (w // 2, int(h * 0.6)),        # Centre
        ]
    elif target == "ground":
        return [
            (w // 2, int(h * 0.85)),       # Bas centre
        ]
        
    # === PORTRAIT ===
    elif target == "face":
        return [
            (w // 2, int(h * 0.35)),       # Centre visage
        ]
    elif target == "hair":
        return [
            (w // 2, int(h * 0.1)),        # Haut tête
        ]
        
    else:
        return [(w // 2, h // 2)]  # Centre par défaut


def _get_default_box(image: Image.Image, target: str, scene_type: str = None) -> tuple:
    """Retourne une box par défaut selon la cible et la scène"""
    w, h = image.size
    
    # === INTÉRIEUR ===
    if target == "floor":
        return (0, int(h * 0.6), w, h)
    elif target == "wall":
        return (0, int(h * 0.1), w, int(h * 0.7))
    elif target == "ceiling":
        return (0, 0, w, int(h * 0.2))
        
    # === ANIMAUX ===
    elif target == "ears":
        return (int(w * 0.15), 0, int(w * 0.85), int(h * 0.3))
    elif target == "eyes":
        return (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.5))
    elif target == "nose":
        return (int(w * 0.3), int(h * 0.4), int(w * 0.7), int(h * 0.7))
    elif target in ["fur", "body"]:
        return (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
        
    # === EXTÉRIEUR ===
    elif target == "sky":
        return (0, 0, w, int(h * 0.4))
    elif target == "ground":
        return (0, int(h * 0.7), w, h)
        
    # === PORTRAIT ===
    elif target == "face":
        return (int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.6))
    elif target == "hair":
        return (int(w * 0.1), 0, int(w * 0.9), int(h * 0.3))
        
    else:
        return (0, 0, w, h)  # Toute l'image par défaut


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


def _save_aerial_metadata(aerial_result: dict, save_path: str = None):
    """
    Sauvegarde les métadonnées de segmentation aérienne pour utilisation ultérieure
    """
    if not save_path:
        return
    
    import json
    import os
    
    # Créer un fichier JSON avec les métadonnées
    metadata_path = save_path.replace(".png", "_metadata.json")
    
    metadata = {
        "elements_found": aerial_result["elements_found"],
        "num_elements": len(aerial_result["elements_found"]),
        "detections": {}
    }
    
    # Ajouter le nombre de détections par élément
    for element_name, detections in aerial_result.get("detections", {}).items():
        metadata["detections"][element_name] = {
            "count": len(detections),
            "avg_score": sum(d["score"] for d in detections) / len(detections) if detections else 0
        }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Métadonnées sauvegardées: {metadata_path}")


def load_aerial_metadata(save_path: str) -> list[str]:
    """
    Charge les métadonnées de segmentation aérienne
    
    Returns:
        Liste des éléments détectés ou None si pas de métadonnées
    """
    import json
    import os
    
    metadata_path = save_path.replace(".png", "_metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return metadata.get("elements_found", [])
    except Exception as e:
        print(f"   ⚠️  Erreur chargement métadonnées: {e}")
        return None
