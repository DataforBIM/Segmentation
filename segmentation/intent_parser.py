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
    action: str  # change_material, change_color, replace, remove, improve, enhance
    target_type: str  # surface, region, instance, global
    target_hint: Optional[str] = None  # floor, wall, sofa, etc.
    material: Optional[str] = None  # marble, wood, etc.
    color: Optional[str] = None  # blue, red, etc.
    style: Optional[str] = None  # modern, luxury, etc.
    scene: Optional[str] = None  # interior, exterior, aerial
    confidence: float = 1.0


# =====================================================
# RÈGLES DE DÉTECTION D'INTENTION
# =====================================================

# Actions et leurs indicateurs
ACTION_KEYWORDS = {
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
    "change_material": "surface",
    "change_color": "region",
    "replace": "instance",
    "remove": "instance",
    "improve": "region",
    "enhance": "region",
    "reconstruct": "region"
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
    
    # 2. Détecter le TYPE DE CIBLE
    target_type = TARGET_TYPE_RULES.get(action, "region")
    
    # 3. Détecter l'INDICE DE CIBLE (floor, wall, sofa, etc.)
    target_hint = _detect_target_hint(prompt_lower)
    
    # Ajuster target_type selon la cible détectée
    if target_hint:
        target_type = _adjust_target_type(target_hint, target_type)
    
    # 4. Détecter le MATÉRIAU
    material = _detect_material(prompt_lower)
    
    # 5. Détecter la COULEUR
    color = _detect_color(prompt_lower)
    
    # 6. Détecter le STYLE
    style = _detect_style(prompt_lower)
    
    # 7. Détecter la SCÈNE
    scene = _detect_scene(prompt_lower)
    
    # 8. Calculer la CONFIANCE
    confidence = _calculate_confidence(action, target_hint, material, color)
    
    return Intent(
        action=action,
        target_type=target_type,
        target_hint=target_hint,
        material=material,
        color=color,
        style=style,
        scene=scene,
        confidence=confidence
    )


def _detect_action(prompt: str) -> str:
    """Détecte l'action principale depuis le prompt"""
    
    # Compter les correspondances pour chaque action
    scores = {}
    for action, keywords in ACTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt)
        if score > 0:
            scores[action] = score
    
    if not scores:
        return "improve"  # Action par défaut
    
    # Retourner l'action avec le plus de correspondances
    return max(scores, key=scores.get)


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
    
    if intent.target_hint:
        parts.append(f"Target: {intent.target_hint}")
    
    parts.append(f"Type: {intent.target_type}")
    
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
        "target_type": intent.target_type,
        "target_hint": intent.target_hint,
        "material": intent.material,
        "color": intent.color,
        "style": intent.style,
        "scene": intent.scene,
        "confidence": intent.confidence
    }
