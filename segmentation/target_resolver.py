# =====================================================
# ÉTAPE 2: TARGET RESOLVER
# =====================================================
# Résout la cible exacte à segmenter selon l'intention
# Détermine quoi segmenter et comment

from dataclasses import dataclass, field
from typing import Optional
from .intent_parser import Intent


@dataclass
class Target:
    """Structure représentant la cible de segmentation"""
    
    # Cibles primaires (à modifier)
    primary: list[str] = field(default_factory=list)
    
    # Cibles à protéger (ne jamais toucher)
    protected: list[str] = field(default_factory=list)
    
    # Cibles de contexte (pour cohérence)
    context: list[str] = field(default_factory=list)
    
    # Méthode de segmentation recommandée
    method: str = "semantic"  # "semantic", "sam2", "hybrid"
    
    # Priorité de chaque cible (pour fusion)
    priorities: dict = field(default_factory=dict)
    
    # Métadonnées
    scene: Optional[str] = None
    confidence: float = 1.0


# =====================================================
# RÈGLES DE RÉSOLUTION DE CIBLES
# =====================================================

# Mapping intention → cibles multiples
INTENT_TO_TARGETS = {
    # Prompts simples avec cible explicite
    "floor": {
        "primary": ["floor"],
        "protected": ["furniture", "person", "object"],
        "context": ["wall"],
        "method": "semantic"
    },
    "wall": {
        "primary": ["wall"],
        "protected": ["furniture", "person", "window", "door", "object"],
        "context": ["floor", "ceiling"],
        "method": "semantic"
    },
    "ceiling": {
        "primary": ["ceiling"],
        "protected": ["light", "person"],
        "context": ["wall"],
        "method": "semantic"
    },
    "roof": {
        "primary": ["roof"],
        "protected": ["window", "chimney"],
        "context": ["wall", "building"],
        "method": "semantic"
    },
    "road": {
        "primary": ["road", "sidewalk"],
        "protected": ["car", "person", "vegetation"],
        "context": ["building"],
        "method": "semantic"
    },
    
    # Instances (SAM2)
    "sofa": {
        "primary": ["sofa"],
        "protected": ["person"],
        "context": ["floor"],
        "method": "sam2"
    },
    "table": {
        "primary": ["table"],
        "protected": ["object", "person"],
        "context": ["floor"],
        "method": "sam2"
    },
    "chair": {
        "primary": ["chair"],
        "protected": ["person"],
        "context": ["floor"],
        "method": "sam2"
    },
    "furniture": {
        "primary": ["furniture", "sofa", "table", "chair", "bed"],
        "protected": ["person"],
        "context": ["floor", "wall"],
        "method": "sam2"
    },
    "car": {
        "primary": ["car"],
        "protected": ["person"],
        "context": ["road"],
        "method": "sam2"
    },
    "window": {
        "primary": ["window"],
        "protected": [],
        "context": ["wall"],
        "method": "hybrid"
    },
    "door": {
        "primary": ["door"],
        "protected": [],
        "context": ["wall"],
        "method": "hybrid"
    },
    "building": {
        "primary": ["building", "wall", "roof"],
        "protected": ["window", "door"],
        "context": ["vegetation", "road"],
        "method": "hybrid"
    },
    "facade": {
        "primary": ["wall", "building"],
        "protected": ["window", "door"],
        "context": ["roof"],
        "method": "semantic"
    },
    "vegetation": {
        "primary": ["vegetation", "tree", "plant"],
        "protected": [],
        "context": ["building", "road"],
        "method": "sam2"
    }
}

# Styles → cibles multiples
STYLE_TO_TARGETS = {
    "luxury": {
        "interior": {
            "primary": ["floor", "wall"],
            "protected": ["furniture", "person", "window", "door"],
            "context": ["ceiling"],
            "method": "semantic"
        },
        "exterior": {
            "primary": ["building", "wall"],
            "protected": ["window", "door"],
            "context": ["vegetation"],
            "method": "semantic"
        }
    },
    "modern": {
        "interior": {
            "primary": ["floor", "wall", "ceiling"],
            "protected": ["furniture", "person"],
            "context": [],
            "method": "semantic"
        },
        "exterior": {
            "primary": ["building"],
            "protected": ["window", "door"],
            "context": ["vegetation"],
            "method": "semantic"
        }
    }
}

# Scènes spéciales
SCENE_DEFAULTS = {
    "interior": {
        "primary": ["floor"],
        "protected": ["furniture", "person"],
        "context": ["wall", "ceiling"],
        "method": "semantic"
    },
    "exterior": {
        "primary": ["building"],
        "protected": ["window", "door", "person"],
        "context": ["vegetation", "road"],
        "method": "semantic"
    },
    "aerial": {
        "primary": ["building", "roof", "wall"],
        "protected": ["window"],
        "context": ["road", "vegetation"],
        "method": "hybrid"
    }
}


# =====================================================
# FONCTION PRINCIPALE
# =====================================================

def resolve_target(intent: Intent) -> Target:
    """
    Résout la cible de segmentation depuis l'intention
    
    Args:
        intent: L'intention parsée depuis le prompt
    
    Returns:
        Target avec cibles primaires, protégées, et contexte
    
    Examples:
        >>> intent = Intent(action='change_material', target_hint='floor')
        >>> target = resolve_target(intent)
        >>> target.primary
        ['floor']
        >>> target.protected
        ['furniture', 'person', 'object']
    """
    
    # Cas 1: Cible explicite dans l'intention
    if intent.target_hint and intent.target_hint in INTENT_TO_TARGETS:
        config = INTENT_TO_TARGETS[intent.target_hint]
        return Target(
            primary=config["primary"].copy(),
            protected=config["protected"].copy(),
            context=config["context"].copy(),
            method=config["method"],
            priorities=_calculate_priorities(config["primary"]),
            scene=intent.scene,
            confidence=intent.confidence
        )
    
    # Cas 2: Style global (luxury, modern, etc.)
    if intent.style and intent.style in STYLE_TO_TARGETS:
        scene = intent.scene or "interior"
        if scene in STYLE_TO_TARGETS[intent.style]:
            config = STYLE_TO_TARGETS[intent.style][scene]
            return Target(
                primary=config["primary"].copy(),
                protected=config["protected"].copy(),
                context=config["context"].copy(),
                method=config["method"],
                priorities=_calculate_priorities(config["primary"]),
                scene=scene,
                confidence=intent.confidence * 0.8  # Moins précis sans cible explicite
            )
    
    # Cas 3: Scène sans cible explicite
    if intent.scene and intent.scene in SCENE_DEFAULTS:
        config = SCENE_DEFAULTS[intent.scene]
        return Target(
            primary=config["primary"].copy(),
            protected=config["protected"].copy(),
            context=config["context"].copy(),
            method=config["method"],
            priorities=_calculate_priorities(config["primary"]),
            scene=intent.scene,
            confidence=intent.confidence * 0.6  # Encore moins précis
        )
    
    # Cas 4: Fallback - Amélioration générale
    return Target(
        primary=["floor"],
        protected=["furniture", "person"],
        context=["wall"],
        method="semantic",
        priorities={"floor": 1.0},
        scene=intent.scene or "interior",
        confidence=0.5
    )


def _calculate_priorities(primary_targets: list[str]) -> dict:
    """Calcule les priorités pour chaque cible"""
    
    # Plus la cible est importante, plus sa priorité est haute
    priority_order = {
        "floor": 1.0,
        "wall": 0.9,
        "ceiling": 0.8,
        "building": 1.0,
        "roof": 0.9,
        "road": 0.8,
        "sofa": 1.0,
        "table": 0.9,
        "chair": 0.8,
        "furniture": 0.9,
        "window": 0.7,
        "door": 0.7
    }
    
    priorities = {}
    for i, target in enumerate(primary_targets):
        base_priority = priority_order.get(target, 0.5)
        # Légère réduction selon la position dans la liste
        priorities[target] = base_priority - (i * 0.05)
    
    return priorities


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def is_surface_target(target_name: str) -> bool:
    """Vérifie si la cible est une surface (segmentation sémantique)"""
    
    surfaces = ["floor", "wall", "ceiling", "roof", "road", "sidewalk", "facade"]
    return target_name in surfaces


def is_instance_target(target_name: str) -> bool:
    """Vérifie si la cible est une instance (SAM2)"""
    
    instances = [
        "sofa", "table", "chair", "bed", "furniture",
        "car", "person", "vegetation", "tree", "plant"
    ]
    return target_name in instances


def is_architectural_target(target_name: str) -> bool:
    """Vérifie si la cible est architecturale (hybride)"""
    
    architectural = ["window", "door", "building", "balcony"]
    return target_name in architectural


def get_segmentation_method(target_name: str) -> str:
    """Retourne la méthode de segmentation recommandée pour une cible"""
    
    if is_surface_target(target_name):
        return "semantic"
    elif is_instance_target(target_name):
        return "sam2"
    elif is_architectural_target(target_name):
        return "hybrid"
    else:
        return "semantic"


def describe_target(target: Target) -> str:
    """Génère une description lisible de la cible"""
    
    parts = []
    
    parts.append(f"Primary: {', '.join(target.primary)}")
    
    if target.protected:
        parts.append(f"Protected: {', '.join(target.protected)}")
    
    if target.context:
        parts.append(f"Context: {', '.join(target.context)}")
    
    parts.append(f"Method: {target.method}")
    parts.append(f"Confidence: {target.confidence:.0%}")
    
    return " | ".join(parts)


def target_to_dict(target: Target) -> dict:
    """Convertit le Target en dictionnaire"""
    
    return {
        "primary": target.primary,
        "protected": target.protected,
        "context": target.context,
        "method": target.method,
        "priorities": target.priorities,
        "scene": target.scene,
        "confidence": target.confidence
    }
