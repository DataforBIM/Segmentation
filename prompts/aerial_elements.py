# Prompts spécifiques pour chaque élément d'une scène aérienne

AERIAL_ELEMENT_PROMPTS = {
    # "building": DÉSACTIVÉ - remplacé par walls + ornementation
    
    "walls": {
        "positive": (
            "detailed exterior wall, "
            "sharp wall texture, "
            "realistic wall materials, "
            "clear wall surface, "
            "photorealistic facade, "
            "high-resolution wall details, "
            "accurate wall proportions, "
            "natural wall weathering"
        ),
        "negative": (
            "blurry wall, "
            "distorted wall, "
            "warped facade, "
            "low resolution wall, "
            "merged walls, "
            "unclear wall boundaries"
        )
    },
    
    "ornementation": {
        "positive": (
            "detailed architectural ornamentation, "
            "sharp decorative elements, "
            "realistic ornamental details, "
            "clear molding and trim, "
            "photorealistic decoration, "
            "high-resolution ornaments, "
            "accurate architectural details, "
            "natural weathering on ornaments"
        ),
        "negative": (
            "blurry ornamentation, "
            "distorted decoration, "
            "pixelated details, "
            "low resolution ornaments, "
            "merged decorative elements, "
            "unclear ornamental boundaries"
        )
    },
    
    "roof": {
        "positive": (
            "detailed roof texture, "
            "realistic roofing materials, "
            "clear roof tiles or shingles, "
            "sharp roof edges, "
            "natural roof weathering, "
            "photorealistic roof surface, "
            "accurate roof shadows"
        ),
        "negative": (
            "blurry roof, "
            "pixelated roof texture, "
            "flat roof without detail, "
            "unrealistic roof material, "
            "distorted roof shape"
        )
    },
    
    "window": {
        "positive": (
            "sharp window details, "
            "realistic glass reflections, "
            "clear window frames, "
            "photorealistic window, "
            "detailed window panes, "
            "accurate window proportions, "
            "natural glass reflections"
        ),
        "negative": (
            "blurry window, "
            "missing window frames, "
            "unrealistic reflections, "
            "distorted window shape, "
            "low resolution glass"
        )
    },
    
    "door": {
        "positive": (
            "detailed door texture, "
            "clear door frame, "
            "realistic door material, "
            "sharp door edges, "
            "photorealistic entrance, "
            "accurate door proportions"
        ),
        "negative": (
            "blurry door, "
            "distorted door shape, "
            "unrealistic door material, "
            "low resolution door"
        )
    },
    
    "road": {
        "positive": (
            "smooth asphalt surface, "
            "realistic road texture, "
            "clear road markings, "
            "photorealistic pavement, "
            "natural road weathering, "
            "accurate road lines, "
            "high-resolution road surface"
        ),
        "negative": (
            "blurry road, "
            "distorted road lines, "
            "unrealistic pavement, "
            "broken road surface, "
            "disconnected road segments"
        )
    },
    
    "road_markings": {
        "positive": (
            "crisp road lines, "
            "sharp road markings, "
            "clear lane markings, "
            "precise white lines, "
            "detailed crosswalk, "
            "photorealistic road paint, "
            "accurate road signs, "
            "high-resolution markings, "
            "well-defined zebra crossing, "
            "clear road direction arrows"
        ),
        "negative": (
            "blurry markings, "
            "faded road lines, "
            "unclear markings, "
            "distorted lines, "
            "missing road paint, "
            "low resolution markings, "
            "broken lines, "
            "irregular markings"
        )
    },
    
    "car": {
        "positive": (
            "detailed car model, "
            "realistic vehicle, "
            "accurate car proportions, "
            "photorealistic automobile, "
            "sharp car details, "
            "natural car colors, "
            "proper car shadows"
        ),
        "negative": (
            "blurry car, "
            "distorted vehicle, "
            "wrong car scale, "
            "unrealistic car proportions, "
            "toy car appearance, "
            "low resolution vehicle"
        )
    },
    
    "vegetation": {
        "positive": (
            "natural tree textures, "
            "realistic vegetation, "
            "detailed foliage, "
            "photorealistic plants, "
            "natural greenery, "
            "accurate tree shapes, "
            "organic vegetation patterns"
        ),
        "negative": (
            "fake trees, "
            "plastic plants, "
            "unrealistic vegetation, "
            "blurry foliage, "
            "artificial greenery, "
            "low resolution trees"
        )
    },
    
    "parking": {
        "positive": (
            "organized parking layout, "
            "clear parking lines, "
            "realistic parking lot surface, "
            "photorealistic parking space, "
            "accurate parking markings, "
            "natural pavement texture"
        ),
        "negative": (
            "distorted parking lines, "
            "unrealistic parking layout, "
            "blurry parking surface, "
            "irregular parking spaces"
        )
    },
    
    "sidewalk": {
        "positive": (
            "clear sidewalk texture, "
            "realistic pavement, "
            "detailed footpath surface, "
            "photorealistic sidewalk, "
            "natural sidewalk materials, "
            "accurate sidewalk edges"
        ),
        "negative": (
            "blurry sidewalk, "
            "distorted footpath, "
            "unrealistic pavement, "
            "low resolution sidewalk"
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
        "door": "portes",
        "road": "routes",
        "road_markings": "marquages routiers",
        "sidewalk": "trottoirs",
        "car": "voitures",
        "vegetation": "végétation",
        "parking": "parkings"
    }
    
    return descriptions.get(element, element)
