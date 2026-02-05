# =====================================================
# ÉTAPE 1: INTENT PARSER
# =====================================================
# Analyse le prompt utilisateur pour comprendre l'intention
# Similaire à la compréhension de ChatGPT

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class Intent:
    """Structure représentant l'intention de l'utilisateur"""
    action: str  # change_material, change_color, replace, remove, improve, enhance, add
    action_type: str  # ADD, MODIFY, REMOVE
    target_type: str  # surface, region, instance, global, spatial_zone
    target_hint: Optional[str] = None  # floor, wall, sofa, etc.
    
    # NOUVEAUX pour actions ADD
    object_to_add: Optional[str] = None  # "flower", "tree", "furniture"
    location: Optional[str] = None  # "garden_foreground", "lawn_edge", "flowerbed_zone"
    
    material: Optional[str] = None  # marble, wood, etc.
    color: Optional[str] = None  # blue, red, etc.
    style: Optional[str] = None  # modern, luxury, etc.
    scene: Optional[str] = None  # interior, exterior, aerial
    confidence: float = 1.0


# =====================================================
# RÈGLES DE DÉTECTION D'INTENTION
# =====================================================

# =====================================================
# CLASSIFICATION D'ACTIONS (ANALYSE SÉMANTIQUE)
# =====================================================

# Liste de référence pour détection (mais pas utilisée directement)
# La classification sera basée sur l'INTENTION, pas sur les mots

# Indicateurs de MODIFICATION (propriété d'un objet existant)
MODIFICATION_INDICATORS = {
    "property_change": ["color", "couleur", "material", "matériau", "texture", "finish", "surface"],
    "transformation": ["to", "en", "into", "vers", "→"],
    "existing_object": ["the", "le", "la", "les", "this", "ce", "cette"]
}

# Indicateurs d'AJOUT (introduire quelque chose de nouveau)
ADDITION_INDICATORS = {
    "new_objects": ["rose", "roses", "flower", "flowers", "fleur", "fleurs", "tree", "arbre", "plant", "plante"],
    "containers": ["in", "dans", "on", "sur", "at", "à", "with", "avec"],
    "quantity": ["some", "few", "plusieurs", "quelques", "a bit", "un peu"]
}

# Indicateurs de SUPPRESSION
REMOVAL_INDICATORS = {
    "actions": ["remove", "delete", "erase", "clear", "supprimer", "effacer", "enlever"],
    "negation": ["without", "sans", "no", "pas de"]
}

# Actions et leurs indicateurs (pour détection legacy)
ACTION_KEYWORDS = {
    "add": [
        "add", "insert", "place", "introduce", "put", "install",
        "ajouter", "insérer", "placer", "introduire", "mettre", "installer",
        "plant", "grow", "planter", "pousser"
    ],
    "change_material": [
        "change", "replace", "convert", "transform",
        "marble", "wood", "tile", "parquet", "concrete", "stone", "brick",
        "material", "texture", "finish", "surface",
        "changer", "remplacer", "convertir", "transformer",
        "marbre", "bois", "carrelage", "béton", "pierre", "brique"
    ],
    "change_color": [
        "paint", "color", "colour", "repaint", "tint", "tone",
        "peindre", "couleur", "repeindre", "teinte",
        "blue", "red", "green", "white", "black", "grey", "gray",
        "bleu", "rouge", "vert", "blanc", "noir", "gris"
    ],
    "replace": [
        "replace", "swap", "substitute", "switch",
        "remplacer", "échanger", "substituer"
    ],
    "remove": [
        "remove", "delete", "erase", "clear", "clean",
        "supprimer", "effacer", "nettoyer", "enlever"
    ],
    "improve": [
        "improve", "enhance", "upgrade", "better", "fix", "correct",
        "améliorer", "corriger", "réparer"
    ],
    "enhance": [
        "enhance", "beautify", "luxury", "luxurious", "modern", "premium",
        "embellir", "luxe", "luxueux", "moderne", "premium"
    ],
    "reconstruct": [
        "reconstruct", "rebuild", "restore", "repair",
        "reconstruire", "restaurer", "réparer"
    ]
}

# Types de cibles selon l'intention
TARGET_TYPE_RULES = {
    # action → target_type par défaut
    "add": "spatial_zone",  # ✨ NOUVEAU: zone d'accueil
    "change_material": "surface",
    "change_color": "region",
    "replace": "instance",
    "remove": "instance",
    "improve": "region",
    "enhance": "region",
    "reconstruct": "region"
}

# OBJETS AJOUTABLES (pour action ADD)
ADDABLE_OBJECTS = {
    "flowers": ["flower", "flowers", "fleur", "fleurs", "rose", "roses", "tulip", "tulipes"],
    "tree": ["tree", "trees", "arbre", "arbres"],
    "plant": ["plant", "plants", "plante", "plantes"],
    "furniture": ["furniture", "meuble", "meubles", "sofa", "table", "chair"],
    "decoration": ["decoration", "décoration", "ornament", "ornement"]
}

# LOCATIONS (zones spatiales)
LOCATION_KEYWORDS = {
    "foreground": ["foreground", "premier plan", "front", "avant", "close", "proche"],
    "midground": ["midground", "milieu", "middle", "center", "centre"],
    "background": ["background", "arrière-plan", "back", "arrière", "far", "loin"],
    "edge": ["edge", "bord", "border", "bordure"],
    "corner": ["corner", "coin"],
    "center": ["center", "centre", "middle", "milieu"]
}

# Indicateurs de cible
TARGET_KEYWORDS = {
    # Surfaces (segmentation sémantique)
    "floor": ["floor", "ground", "sol", "plancher", "parquet", "tile", "carrelage"],
    "wall": ["wall", "walls", "mur", "murs", "facade", "façade"],
    "ceiling": ["ceiling", "plafond"],
    "roof": ["roof", "rooftop", "toit", "toiture"],
    "road": ["road", "street", "pavement", "route", "rue", "trottoir", "sidewalk"],
    
    # Instances (SAM2)
    "sofa": ["sofa", "couch", "canapé", "settee"],
    "table": ["table", "desk", "bureau"],
    "chair": ["chair", "seat", "chaise", "siège"],
    "bed": ["bed", "lit"],
    "window": ["window", "fenêtre", "vitrage", "glazing"],
    "door": ["door", "porte", "entrance", "entrée"],
    "furniture": ["furniture", "meuble", "meubles", "mobilier"],
    "car": ["car", "vehicle", "voiture", "véhicule"],
    "person": ["person", "people", "personne", "gens", "human"],
    "vegetation": ["vegetation", "tree", "plant", "végétation", "arbre", "plante"],
    "building": ["building", "bâtiment", "immeuble", "structure"]
}

# Indicateurs de scène
SCENE_KEYWORDS = {
    "interior": ["interior", "indoor", "room", "inside", "intérieur", "pièce", "salon", "chambre"],
    "exterior": ["exterior", "outdoor", "outside", "extérieur", "dehors"],
    "aerial": ["aerial", "bird", "overhead", "top-down", "aérien", "vue aérienne", "drone"]
}

# Matériaux courants
MATERIALS = [
    "marble", "marbre", "wood", "bois", "tile", "carrelage", "concrete", "béton",
    "stone", "pierre", "brick", "brique", "metal", "métal", "glass", "verre",
    "parquet", "carpet", "tapis", "laminate", "vinyl", "ceramic", "céramique",
    "granite", "terrazzo", "epoxy", "resin", "résine"
]

# Couleurs courantes
COLORS = [
    "white", "blanc", "black", "noir", "grey", "gray", "gris",
    "blue", "bleu", "red", "rouge", "green", "vert", "yellow", "jaune",
    "orange", "purple", "violet", "pink", "rose", "brown", "marron", "brun",
    "beige", "cream", "crème", "navy", "turquoise", "gold", "or", "silver", "argent"
]

# Styles
STYLES = [
    "modern", "moderne", "contemporary", "contemporain",
    "luxury", "luxe", "luxurious", "luxueux",
    "minimalist", "minimaliste", "classic", "classique",
    "industrial", "industriel", "rustic", "rustique",
    "scandinavian", "scandinave", "bohemian", "bohème"
]


def parse_intent(prompt: str) -> Intent:
    """
    Analyse le prompt utilisateur et extrait l'intention
    
    Args:
        prompt: Le prompt de l'utilisateur
    
    Returns:
        Intent avec action, target_type, et autres infos détectées
    
    Examples:
        >>> parse_intent("change floor to marble")
        Intent(action='change_material', target_type='surface', target_hint='floor', material='marble')
        
        >>> parse_intent("replace the sofa")
        Intent(action='replace', target_type='instance', target_hint='sofa')
        
        >>> parse_intent("luxury interior design")
        Intent(action='enhance', target_type='region', style='luxury', scene='interior')
    """
    
    prompt_lower = prompt.lower()
    
    # 1. Détecter l'ACTION
    action = _detect_action(prompt_lower)
    
    # 2. Classifier le TYPE D'ACTION (ADD/MODIFY/REMOVE)
    action_type = _classify_action_type(action, prompt_lower)
    
    # 3. Détecter le TYPE DE CIBLE
    target_type = TARGET_TYPE_RULES.get(action, "region")
    
    # 4. Détecter l'INDICE DE CIBLE (floor, wall, sofa, etc.)
    target_hint = _detect_target_hint(prompt_lower)
    
    # 5. NOUVEAUX: Pour actions ADD, détecter objet + location
    object_to_add = None
    location = None
    
    if action_type == "ADD":
        object_to_add = _detect_object_to_add(prompt_lower)
        location = _detect_location(prompt_lower, target_hint)
    
    # Ajuster target_type selon la cible détectée
    if target_hint:
        target_type = _adjust_target_type(target_hint, target_type)
    
    # 6. Détecter le MATÉRIAU
    material = _detect_material(prompt_lower)
    
    # 7. Détecter la COULEUR
    color = _detect_color(prompt_lower)
    
    # 8. Détecter le STYLE
    style = _detect_style(prompt_lower)
    
    # 9. Détecter la SCÈNE
    scene = _detect_scene(prompt_lower)
    
    # 10. Calculer la CONFIANCE
    confidence = _calculate_confidence(action, target_hint, material, color)
    
    return Intent(
        action=action,
        action_type=action_type,
        target_type=target_type,
        target_hint=target_hint,
        object_to_add=object_to_add,
        location=location,
        material=material,
        color=color,
        style=style,
        scene=scene,
        confidence=confidence
    )


def _detect_action(prompt: str) -> str:
    """Détecte l'action principale depuis le prompt (legacy)"""
    
    # Cette fonction reste pour compatibilité
    # La vraie classification se fait dans _classify_action_type
    
    scores = {}
    for action, keywords in ACTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt)
        if score > 0:
            scores[action] = score
    
    if not scores:
        return "improve"  # Action par défaut
    
    return max(scores, key=scores.get)


def _classify_action_type(action: str, prompt: str) -> str:
    """
    Classifie l'action en ADD/MODIFY/REMOVE basé sur l'INTENTION SÉMANTIQUE
    
    LOGIQUE (ordre important):
    1. REMOVE: Suppression explicite (remove, delete)
    2. ADD: Introduction d'un nouvel objet avec contexte spatial
    3. MODIFY: Changement de propriété d'un objet existant (par défaut)
    
    Args:
        action: Action détectée (legacy)
        prompt: Prompt utilisateur
    
    Returns:
        "ADD", "MODIFY", ou "REMOVE"
    """
    
    # ============================================
    # 1. REMOVE - Le plus explicite
    # ============================================
    if _is_removal_intent(prompt):
        return "REMOVE"
    
    # ============================================
    # 2. ADD - Introduction d'un nouvel élément
    # ============================================
    # IMPORTANT: Vérifier ADD avant MODIFY car plus spécifique
    if _is_addition_intent(prompt):
        return "ADD"
    
    # ============================================
    # 3. MODIFY - Changement de propriété (défaut)
    # ============================================
    return "MODIFY"


def _is_removal_intent(prompt: str) -> bool:
    """Détecte une intention de suppression"""
    
    # Indicateurs explicites
    removal_words = ["remove", "delete", "erase", "clear", 
                     "supprimer", "effacer", "enlever", "retirer"]
    
    return any(word in prompt for word in removal_words)


def _is_modification_intent(prompt: str) -> bool:
    """
    Détecte une intention de modification
    
    Signes:
    - Changement de propriété: "change color", "changer la couleur"
    - Transformation: "to marble", "en marbre"
    - Référence à objet existant: "the floor", "le mur"
    """
    
    # Structure typique: [change/modifier] + [cible existante] + [to/en] + [nouvelle propriété]
    
    has_property_change = any(
        prop in prompt 
        for indicators in MODIFICATION_INDICATORS.values() 
        for prop in indicators
    )
    
    # Si on mentionne une transformation (to, en, into)
    has_transformation = any(word in prompt for word in ["to ", " en ", "into", "→"])
    
    # Si on mentionne un article défini (the, le, la)
    has_existing_ref = any(word in prompt for word in ["the ", "le ", "la ", "les ", "this "])
    
    # MODIFICATION = changement de propriété OU transformation d'existant
    return has_property_change or (has_transformation and has_existing_ref)


def _is_addition_intent(prompt: str) -> bool:
    """
    Détecte une intention d'ajout
    
    Signes clés (TOUS requis pour haute confiance):
    1. Nouveaux objets à introduire (roses, fleurs, arbres)
    2. Contexte spatial (in, dans, on, sur) OU quantificateur (un peu, quelques)
    3. ABSENCE de transformation explicite (to X, en X)
    
    Exemples:
    - "Ajouter des roses dans le jardin" → ADD (objet + lieu)
    - "Le jardin avec des roses" → ADD (objet + contexte)
    - "Quelques fleurs" → ADD (quantificateur + objet)
    - "Changer le sol en marbre" → MODIFY (transformation)
    """
    
    prompt_lower = prompt.lower()
    
    # 1. OBJETS À INTRODUIRE (nouveaux éléments)
    new_objects = [
        "rose", "roses", "flower", "flowers", "fleur", "fleurs",
        "tree", "trees", "arbre", "arbres",
        "plant", "plants", "plante", "plantes",
        "furniture", "meuble", "decoration", "décoration",
        "végétation", "vegetation"
    ]
    
    has_new_object = any(obj in prompt_lower for obj in new_objects)
    
    if not has_new_object:
        return False  # Pas d'objet à ajouter
    
    # 2. CONTEXTE SPATIAL ou QUANTIFICATEUR
    # Prépositions de lieu
    location_patterns = [
        " in ", " dans ", " on ", " sur ", " at ", " à ",
        " with ", " avec ",  # "jardin avec des roses"
        " near ", " près ",  # "près de l'entrée"
    ]
    has_location = any(loc in prompt_lower for loc in location_patterns)
    
    # Quantificateurs (quelques, un peu, des)
    quantity_patterns = [
        "some ", "few ", "plusieurs ", "quelques ", 
        "a bit", "un peu", "bit of", "des ",
        "a few", "un peu de"
    ]
    has_quantity = any(q in prompt_lower for q in quantity_patterns)
    
    # 3. VÉRIFIER ABSENCE DE TRANSFORMATION EXPLICITE
    # Si on dit "en marbre" ou "to marble", c'est une MODIFICATION
    transformation_patterns = [" to ", " en ", " into ", "→"]
    has_explicit_transformation = any(t in prompt_lower for t in transformation_patterns)
    
    # Si transformation explicite vers un matériau/couleur, c'est MODIFY
    materials = ["marble", "marbre", "wood", "bois", "concrete", "béton", "glass", "verre"]
    colors = ["white", "blanc", "blue", "bleu", "red", "rouge", "black", "noir"]
    
    if has_explicit_transformation:
        # Vérifier si transformation vers matériau/couleur
        has_material_target = any(mat in prompt_lower for mat in materials)
        has_color_target = any(col in prompt_lower for col in colors)
        
        if has_material_target or has_color_target:
            return False  # C'est une MODIFICATION, pas un ajout
    
    # DÉCISION FINALE:
    # ADD si: objet nouveau + (contexte spatial OU quantificateur) ET pas de transformation matériau
    return has_new_object and (has_location or has_quantity)


def _detect_object_to_add(prompt: str) -> Optional[str]:
    """Détecte l'objet à ajouter (pour actions ADD)"""
    
    for obj_type, keywords in ADDABLE_OBJECTS.items():
        if any(kw in prompt for kw in keywords):
            return obj_type
    
    return None


def _detect_location(prompt: str, target_hint: Optional[str]) -> Optional[str]:
    """Détecte la location/zone spatiale (pour actions ADD)"""
    
    # Détecter position spatiale
    position = None
    for pos_name, keywords in LOCATION_KEYWORDS.items():
        if any(kw in prompt for kw in keywords):
            position = pos_name
            break
    
    # Construire location complète: {contexte}_{position}
    context = target_hint if target_hint else "ground"
    
    if position:
        return f"{context}_{position}"
    else:
        return f"{context}_foreground"  # Défaut


def _detect_target_hint(prompt: str) -> Optional[str]:
    """Détecte l'indice de cible depuis le prompt"""
    
    for target, keywords in TARGET_KEYWORDS.items():
        if any(kw in prompt for kw in keywords):
            return target
    
    return None


def _adjust_target_type(target_hint: str, default_type: str) -> str:
    """Ajuste le type de cible selon l'indice détecté"""
    
    # Surfaces → segmentation sémantique
    surfaces = ["floor", "wall", "ceiling", "roof", "road"]
    if target_hint in surfaces:
        return "surface"
    
    # Instances → SAM2
    instances = ["sofa", "table", "chair", "bed", "car", "person", "furniture"]
    if target_hint in instances:
        return "instance"
    
    # Éléments architecturaux → hybride
    architectural = ["window", "door", "building"]
    if target_hint in architectural:
        return "architectural"
    
    return default_type


def _detect_material(prompt: str) -> Optional[str]:
    """Détecte le matériau mentionné"""
    
    for material in MATERIALS:
        if material in prompt:
            return material
    return None


def _detect_color(prompt: str) -> Optional[str]:
    """Détecte la couleur mentionnée"""
    
    for color in COLORS:
        if color in prompt:
            return color
    return None


def _detect_style(prompt: str) -> Optional[str]:
    """Détecte le style mentionné"""
    
    for style in STYLES:
        if style in prompt:
            return style
    return None


def _detect_scene(prompt: str) -> Optional[str]:
    """Détecte le type de scène"""
    
    for scene, keywords in SCENE_KEYWORDS.items():
        if any(kw in prompt for kw in keywords):
            return scene
    return None


def _calculate_confidence(action: str, target_hint: str, material: str, color: str) -> float:
    """Calcule un score de confiance pour l'intention détectée"""
    
    confidence = 0.5  # Base
    
    if action != "improve":  # Action spécifique détectée
        confidence += 0.2
    
    if target_hint:  # Cible détectée
        confidence += 0.2
    
    if material or color:  # Détail spécifique
        confidence += 0.1
    
    return min(confidence, 1.0)


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def describe_intent(intent: Intent) -> str:
    """Génère une description lisible de l'intention"""
    
    parts = [f"Action: {intent.action}"]
    parts.append(f"Type: {intent.action_type}")  # ✨ NOUVEAU
    
    if intent.target_hint:
        parts.append(f"Target: {intent.target_hint}")
    
    parts.append(f"TargetType: {intent.target_type}")
    
    # ✨ NOUVEAUX champs pour ADD
    if intent.action_type == "ADD":
        if intent.object_to_add:
            parts.append(f"Object: {intent.object_to_add}")
        if intent.location:
            parts.append(f"Location: {intent.location}")
    
    if intent.material:
        parts.append(f"Material: {intent.material}")
    
    if intent.color:
        parts.append(f"Color: {intent.color}")
    
    if intent.style:
        parts.append(f"Style: {intent.style}")
    
    if intent.scene:
        parts.append(f"Scene: {intent.scene}")
    
    parts.append(f"Confidence: {intent.confidence:.0%}")
    
    return " | ".join(parts)


def intent_to_dict(intent: Intent) -> dict:
    """Convertit l'Intent en dictionnaire"""
    
    return {
        "action": intent.action,
        "action_type": intent.action_type,  # ✨ NOUVEAU
        "target_type": intent.target_type,
        "target_hint": intent.target_hint,
        "object_to_add": intent.object_to_add,  # ✨ NOUVEAU
        "location": intent.location,  # ✨ NOUVEAU
        "material": intent.material,
        "color": intent.color,
        "style": intent.style,
        "scene": intent.scene,
        "confidence": intent.confidence
    }
