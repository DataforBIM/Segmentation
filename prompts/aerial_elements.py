# Prompts spécifiques pour chaque élément d'une scène aérienne

# ============================================================
# AERIAL SCENE PROMPTS – ARCHITECTURAL SDXL PIPELINE
# Authoritative prompts for SDXL + ControlNet + SAM2
# ============================================================

# ------------------------------------------------------------------
# ELEMENT-LEVEL PROMPTS (USED WITH SEGMENTATION MASKS)
# ------------------------------------------------------------------

AERIAL_ELEMENT_PROMPTS = {

    # =========================
    # STRUCTURE (MAX AUTHORITY)
    # =========================

    "walls": {
        "positive": (
            "preserve existing exterior walls, "
            "maintain original wall geometry, "
            "straightened and corrected wall planes, "
            "architectural wall surfaces clearly defined, "
            "accurate wall proportions, "
            "physically plausible wall materials, "
            "natural facade weathering, "
            "no geometry simplification"
        ),
        "negative": (
            "removed walls, "
            "merged walls, "
            "warped facade, "
            "bent geometry, "
            "melted surfaces, "
            "flattened architecture, "
            "blurred wall structure"
        )
    },

    "roof": {
        "positive": (
            "preserve original roof shape and volume, "
            "accurate roof geometry, "
            "clean roof planes, "
            "realistic roof materials and tiles, "
            "correct roof edges and ridgelines, "
            "natural roof weathering"
        ),
        "negative": (
            "warped roof geometry, "
            "collapsed roof, "
            "flattened roof volume, "
            "melted tiles, "
            "unrealistic roof materials"
        )
    },

    # =========================
    # OPENINGS (PROTECTED)
    # =========================

    "window": {
        "positive": (
            "preserve all existing windows, "
            "do not remove or relocate windows, "
            "maintain original window positions, "
            "architectural openings clearly defined, "
            "windows as structural facade elements, "
            "consistent facade fenestration, "
            "sharp window frames, "
            "realistic glass behavior"
        ),
        "negative": (
            "removed windows, "
            "missing windows, "
            "filled openings, "
            "smoothed over windows, "
            "sealed facade, "
            "fake or hallucinated windows, "
            "distorted window geometry"
        )
    },

    "door": {
        "positive": (
            "preserve existing doors, "
            "maintain original entrance positions, "
            "clear door openings, "
            "accurate door proportions, "
            "architectural entrance consistency"
        ),
        "negative": (
            "removed doors, "
            "missing entrances, "
            "sealed door openings, "
            "distorted door geometry"
        )
    },

    # =========================
    # SECONDARY DETAILS
    # =========================

    "ornementation": {
        "positive": (
            "preserve existing architectural ornamentation, "
            "maintain original decorative placement, "
            "clear and readable molding and trims, "
            "architectural details without exaggeration, "
            "natural material aging"
        ),
        "negative": (
            "removed ornamentation, "
            "over-simplified decoration, "
            "exaggerated details, "
            "decorative noise, "
            "merged decorative elements"
        )
    },

    # =========================
    # INFRASTRUCTURE (LOW PRIORITY)
    # =========================

    "road": {
        "positive": (
            "clean road surface, "
            "realistic asphalt texture, "
            "consistent road layout, "
            "natural wear without distraction"
        ),
        "negative": (
            "broken road continuity, "
            "floating road segments, "
            "overly sharp textures, "
            "dominant road features"
        )
    },

    "road_markings": {
        "positive": (
            "preserve existing road markings, "
            "clear but subtle lane markings, "
            "consistent crosswalk patterns"
        ),
        "negative": (
            "exaggerated markings, "
            "overly bright paint, "
            "dominant road graphics"
        )
    },

    "sidewalk": {
        "positive": (
            "clear sidewalk layout, "
            "realistic pavement materials, "
            "consistent sidewalk edges"
        ),
        "negative": (
            "distorted sidewalks, "
            "floating pavement, "
            "dominant sidewalk texture"
        )
    },

    "parking": {
        "positive": (
            "organized parking layout, "
            "subtle parking markings, "
            "non-dominant parking surface"
        ),
        "negative": (
            "chaotic parking layout, "
            "overly strong markings, "
            "dominant parking features"
        )
    },

    # =========================
    # ELEMENTS TO NEUTRALIZE
    # =========================

    "car": {
        "positive": (
            "static parked vehicles, "
            "minimal visual importance, "
            "non-dominant presence"
        ),
        "negative": (
            "focus on cars, "
            "detailed vehicles, "
            "shiny cars, "
            "dominant automobiles, "
            "moving vehicles"
        )
    },

    "vegetation": {
        "positive": (
            "background vegetation only, "
            "soft natural greenery, "
            "non-dominant trees and plants"
        ),
        "negative": (
            "overgrown vegetation, "
            "dominant trees, "
            "blocking architecture, "
            "oversaturated greenery"
        )
    }
}



def build_aerial_prompt(user_prompt: str, elements_found: list[str]) -> tuple[str, str]:
    """
    Construit un prompt enrichi pour les scènes aériennes en tenant compte
    des éléments détectés par la segmentation
    
    Args:
        user_prompt: Prompt de base de l'utilisateur
        elements_found: Liste des éléments détectés (building, roof, etc.)
    
    Returns:
        (prompt_positif_enrichi, prompt_négatif_enrichi)
    """
    # Base du prompt utilisateur
    positive = user_prompt
    
    # Au lieu d'empiler tous les prompts, on fait un résumé général des éléments détectés
    # pour éviter la surcharge et les conflits de prompts
    if elements_found:
        elements_desc = ", ".join(elements_found)
        positive += f", enhance {elements_desc} with realistic details"
    
    # Construire un prompt négatif général plutôt que d'empiler tous les négatifs
    negative = (
        "artificial colors, yellow tint, color cast, wrong colors, "
        "blurry elements, low quality details, "
        "unrealistic materials, fake textures"
    )
    
    return positive, negative


def get_element_description(element: str) -> str:
    """
    Retourne une description lisible d'un élément aérien
    """
    descriptions = {
        # "building": "bâtiments",  # DÉSACTIVÉ
        "walls": "murs",
        "ornementation": "ornementations",
        "roof": "toits",
        "window": "fenêtres",
        "door": "portes",
        "road": "routes",
        "road_markings": "marquages routiers",
        "sidewalk": "trottoirs",
        "car": "voitures",
        "vegetation": "végétation",
        "parking": "parkings"
    }
    
    return descriptions.get(element, element)
