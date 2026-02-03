# Détection automatique de la cible de segmentation depuis le prompt
import re


def detect_segment_target(user_prompt: str) -> str:
    """
    Détecte automatiquement la cible de segmentation depuis le prompt utilisateur
    
    Args:
        user_prompt: Le prompt de l'utilisateur
    
    Returns:
        Target détecté: "floor", "wall", "ceiling", "custom"
    """
    
    prompt_lower = user_prompt.lower()
    
    # Mots-clés pour le sol
    floor_keywords = [
        "floor", "flooring", "ground", "sol", 
        "tile", "tiling", "parquet", "carrelage",
        "marble floor", "wood floor", "concrete floor",
        "carpet", "rug", "tapis"
    ]
    
    # Mots-clés pour les murs
    wall_keywords = [
        "wall", "walls", "mur", "murs",
        "paint wall", "wallpaper", "papier peint",
        "wall texture", "wall color"
    ]
    
    # Mots-clés pour le plafond
    ceiling_keywords = [
        "ceiling", "plafond",
        "ceiling paint", "ceiling color"
    ]
    
    # Compter les occurrences
    floor_count = sum(1 for kw in floor_keywords if kw in prompt_lower)
    wall_count = sum(1 for kw in wall_keywords if kw in prompt_lower)
    ceiling_count = sum(1 for kw in ceiling_keywords if kw in prompt_lower)
    
    # Déterminer la cible dominante
    if floor_count > 0 and floor_count >= wall_count and floor_count >= ceiling_count:
        return "floor"
    elif wall_count > 0 and wall_count >= ceiling_count:
        return "wall"
    elif ceiling_count > 0:
        return "ceiling"
    else:
        # Par défaut, si aucun mot-clé détecté, utiliser floor
        return "floor"


def get_target_description(target: str) -> str:
    """
    Retourne une description lisible de la cible
    """
    descriptions = {
        "floor": "Sol / Floor",
        "wall": "Murs / Walls",
        "ceiling": "Plafond / Ceiling",
        "custom": "Personnalisé / Custom"
    }
    return descriptions.get(target, target)
