# Détection automatique de la cible de segmentation depuis le prompt
import re


# Mapping des cibles valides par type de scène
SCENE_TARGETS = {
    "ANIMAL": ["ears", "eyes", "fur", "tail", "paws", "nose", "body", "background"],
    "INTERIOR": ["floor", "wall", "ceiling", "furniture", "window", "door", "background"],
    "EXTERIOR": ["sky", "ground", "vegetation", "building", "road", "background"],
    "PRODUCT": ["product", "background", "shadow"],
    "PORTRAIT": ["face", "hair", "eyes", "lips", "skin", "clothing", "background"],
    "UNKNOWN": ["floor", "wall", "background"]
}


def detect_segment_target(user_prompt: str, scene_type: str = None) -> str:
    """
    Détecte automatiquement la cible de segmentation depuis le prompt utilisateur
    
    Args:
        user_prompt: Le prompt de l'utilisateur
        scene_type: Type de scène détecté (ANIMAL, INTERIOR, EXTERIOR, etc.)
    
    Returns:
        Target détecté selon la scène
    """
    
    prompt_lower = user_prompt.lower()
    
    # === PARTIES D'ANIMAUX ===
    
    # Mots-clés pour les oreilles
    ears_keywords = [
        "ear", "ears", "oreille", "oreilles",
        "cat ears", "dog ears", "animal ears"
    ]
    
    # Mots-clés pour les yeux
    eyes_keywords = [
        "eye", "eyes", "oeil", "yeux",
        "cat eyes", "dog eyes", "eye color"
    ]
    
    # Mots-clés pour la fourrure/pelage
    fur_keywords = [
        "fur", "fourrure", "pelage", "coat",
        "hair", "poil", "poils", "cheveux",
        "fur color", "coat color"
    ]
    
    # Mots-clés pour la queue
    tail_keywords = [
        "tail", "queue", "cat tail", "dog tail"
    ]
    
    # Mots-clés pour les pattes
    paws_keywords = [
        "paw", "paws", "patte", "pattes",
        "leg", "legs", "jambe", "jambes",
        "foot", "feet", "pied", "pieds"
    ]
    
    # Mots-clés pour le museau/nez
    nose_keywords = [
        "nose", "nez", "muzzle", "museau",
        "snout", "truffe"
    ]
    
    # Mots-clés pour le corps entier
    body_keywords = [
        "body", "corps", "animal", "chat", "cat", "dog", "chien"
    ]
    
    # === ÉLÉMENTS ARCHITECTURAUX (INTERIOR) ===
    
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
    
    # Mots-clés pour les meubles
    furniture_keywords = [
        "furniture", "meuble", "meubles", "sofa", "canapé",
        "table", "chair", "chaise", "bed", "lit", "desk", "bureau"
    ]
    
    # Mots-clés pour les fenêtres
    window_keywords = [
        "window", "windows", "fenêtre", "fenêtres", "vitre"
    ]
    
    # === ÉLÉMENTS EXTÉRIEURS ===
    
    # Mots-clés pour le ciel
    sky_keywords = [
        "sky", "ciel", "clouds", "nuages", "sunset", "sunrise"
    ]
    
    # Mots-clés pour la végétation
    vegetation_keywords = [
        "tree", "trees", "arbre", "arbres", "grass", "herbe",
        "plant", "plants", "plante", "vegetation", "végétation",
        "garden", "jardin", "forest", "forêt"
    ]
    
    # Mots-clés pour les bâtiments
    building_keywords = [
        "building", "bâtiment", "house", "maison", "façade", "facade"
    ]
    
    # Mots-clés pour la route
    road_keywords = [
        "road", "route", "street", "rue", "pavement", "trottoir"
    ]
    
    # === PORTRAIT ===
    
    face_keywords = [
        "face", "visage", "skin", "peau"
    ]
    
    hair_keywords = [
        "hair", "cheveux", "coiffure", "hairstyle"
    ]
    
    lips_keywords = [
        "lips", "lèvres", "mouth", "bouche"
    ]
    
    clothing_keywords = [
        "clothing", "vêtement", "clothes", "shirt", "dress", "robe"
    ]
    
    # === PRODUCT ===
    
    product_keywords = [
        "product", "produit", "item", "objet"
    ]
    
    background_keywords = [
        "background", "fond", "arrière-plan", "arriere-plan",
        "derrière", "derriere", "behind", "backdrop"
    ]
    
    # Compter les occurrences pour chaque catégorie
    counts = {
        # Animaux
        "ears": sum(1 for kw in ears_keywords if kw in prompt_lower),
        "eyes": sum(1 for kw in eyes_keywords if kw in prompt_lower),
        "fur": sum(1 for kw in fur_keywords if kw in prompt_lower),
        "tail": sum(1 for kw in tail_keywords if kw in prompt_lower),
        "paws": sum(1 for kw in paws_keywords if kw in prompt_lower),
        "nose": sum(1 for kw in nose_keywords if kw in prompt_lower),
        "body": sum(1 for kw in body_keywords if kw in prompt_lower),
        # Intérieur
        "floor": sum(1 for kw in floor_keywords if kw in prompt_lower),
        "wall": sum(1 for kw in wall_keywords if kw in prompt_lower),
        "ceiling": sum(1 for kw in ceiling_keywords if kw in prompt_lower),
        "furniture": sum(1 for kw in furniture_keywords if kw in prompt_lower),
        "window": sum(1 for kw in window_keywords if kw in prompt_lower),
        # Extérieur
        "sky": sum(1 for kw in sky_keywords if kw in prompt_lower),
        "vegetation": sum(1 for kw in vegetation_keywords if kw in prompt_lower),
        "building": sum(1 for kw in building_keywords if kw in prompt_lower),
        "road": sum(1 for kw in road_keywords if kw in prompt_lower),
        # Portrait
        "face": sum(1 for kw in face_keywords if kw in prompt_lower),
        "hair": sum(1 for kw in hair_keywords if kw in prompt_lower),
        "lips": sum(1 for kw in lips_keywords if kw in prompt_lower),
        "clothing": sum(1 for kw in clothing_keywords if kw in prompt_lower),
        # Product
        "product": sum(1 for kw in product_keywords if kw in prompt_lower),
        "background": sum(1 for kw in background_keywords if kw in prompt_lower),
    }
    
    # Filtrer selon la scène si fournie
    if scene_type and scene_type in SCENE_TARGETS:
        valid_targets = SCENE_TARGETS[scene_type]
        filtered_counts = {k: v for k, v in counts.items() if k in valid_targets}
    else:
        filtered_counts = counts
    
    # PRIORITÉ: Si "background" est mentionné, c'est toujours prioritaire
    if filtered_counts.get("background", 0) > 0:
        return "background"
    
    # Trouver la catégorie dominante
    max_count = max(filtered_counts.values()) if filtered_counts else 0
    
    if max_count > 0:
        for target, count in filtered_counts.items():
            if count == max_count:
                return target
    
    # Par défaut selon la scène
    return get_default_target_for_scene(scene_type)


def get_default_target_for_scene(scene_type: str) -> str:
    """Retourne la cible par défaut selon le type de scène"""
    defaults = {
        "ANIMAL": "fur",      # Par défaut: tout l'animal
        "INTERIOR": "floor",  # Par défaut: le sol
        "EXTERIOR": "sky",    # Par défaut: le ciel
        "PRODUCT": "product", # Par défaut: le produit
        "PORTRAIT": "face",   # Par défaut: le visage
        "UNKNOWN": "background"
    }
    return defaults.get(scene_type, "floor")


def get_target_description(target: str) -> str:
    """
    Retourne une description lisible de la cible
    """
    descriptions = {
        # Parties d'animaux
        "ears": "Oreilles / Ears",
        "eyes": "Yeux / Eyes",
        "fur": "Fourrure / Fur",
        "tail": "Queue / Tail",
        "paws": "Pattes / Paws",
        "nose": "Museau / Nose",
        "body": "Corps entier / Full body",
        # Intérieur
        "floor": "Sol / Floor",
        "wall": "Murs / Walls",
        "ceiling": "Plafond / Ceiling",
        "furniture": "Meubles / Furniture",
        "window": "Fenêtres / Windows",
        "door": "Portes / Doors",
        # Extérieur
        "sky": "Ciel / Sky",
        "ground": "Sol extérieur / Ground",
        "vegetation": "Végétation / Vegetation",
        "building": "Bâtiments / Buildings",
        "road": "Route / Road",
        # Portrait
        "face": "Visage / Face",
        "hair": "Cheveux / Hair",
        "lips": "Lèvres / Lips",
        "skin": "Peau / Skin",
        "clothing": "Vêtements / Clothing",
        # Product
        "product": "Produit / Product",
        "background": "Arrière-plan / Background",
        # Autres
        "custom": "Personnalisé / Custom"
    }
    return descriptions.get(target, target)
