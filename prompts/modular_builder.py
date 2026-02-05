# =====================================================
# MODULAR PROMPT BUILDER
# prompts/modular_builder.py
# =====================================================
# Construit des prompts en assemblant des modules:
# [SCENE STRUCTURE] + [SUBJECT] + [ENVIRONMENT] + 
# [CAMERA/LENS] + [LIGHTING] + [MATERIALS] + [STYLE]

from prompts.modular_structure import (
    SCENE_STRUCTURES,
    SUBJECTS,
    ENVIRONMENTS,
    CAMERA_SETTINGS,
    LIGHTING_CONDITIONS,
    MATERIALS,
    STYLES,
    get_full_negative_prompt
)


class PromptConfig:
    """Configuration pour la construction de prompts modulaires"""
    
    def __init__(self):
        # Valeurs par défaut
        self.user_prompt = ""
        self.scene_structure = "exterior"
        self.subject = "building"
        self.environment = "urban"
        self.camera = ["eye_level", "straight_verticals"]
        self.lighting = "natural_daylight"
        self.materials = ["mixed_materials", "realistic_weathering"]
        self.style = ["photorealistic", "architectural_photo", "high_quality"]
        self.custom_positive = []
        self.custom_negative = []
    
    def set_user_prompt(self, prompt: str):
        """Définit le prompt utilisateur"""
        self.user_prompt = prompt
        return self
    
    def set_scene_structure(self, structure: str):
        """Définit la structure de scène: interior, exterior, aerial, landscape, detail"""
        self.scene_structure = structure
        return self
    
    def set_subject(self, subject: str):
        """Définit le sujet: building, facade, interior_space, urban_block, etc."""
        self.subject = subject
        return self
    
    def set_environment(self, environment: str):
        """Définit l'environnement: urban, residential, park, street, etc."""
        self.environment = environment
        return self
    
    def set_camera(self, camera: list[str] | str):
        """Définit les paramètres caméra (peut être une liste)"""
        if isinstance(camera, str):
            self.camera = [camera]
        else:
            self.camera = camera
        return self
    
    def set_lighting(self, lighting: str):
        """Définit l'éclairage: natural_daylight, golden_hour, overcast, etc."""
        self.lighting = lighting
        return self
    
    def set_materials(self, materials: list[str] | str):
        """Définit les matériaux (peut être une liste)"""
        if isinstance(materials, str):
            self.materials = [materials]
        else:
            self.materials = materials
        return self
    
    def set_style(self, style: list[str] | str):
        """Définit le style (peut être une liste)"""
        if isinstance(style, str):
            self.style = [style]
        else:
            self.style = style
        return self
    
    def add_custom_positive(self, text: str):
        """Ajoute du texte personnalisé au prompt positif"""
        self.custom_positive.append(text)
        return self
    
    def add_custom_negative(self, text: str):
        """Ajoute du texte personnalisé au prompt négatif"""
        self.custom_negative.append(text)
        return self


def build_modular_prompt(config: PromptConfig) -> tuple[str, str]:
    """
    Construit un prompt complet à partir d'une configuration modulaire
    
    Args:
        config: Configuration du prompt (PromptConfig)
    
    Returns:
        (prompt_positif, prompt_négatif)
    """
    
    # Construction du prompt positif
    prompt_parts = []
    
    # 1. PROMPT UTILISATEUR (priorité maximale)
    if config.user_prompt:
        prompt_parts.append(config.user_prompt)
    
    # 2. SCENE STRUCTURE
    if config.scene_structure in SCENE_STRUCTURES:
        prompt_parts.append(SCENE_STRUCTURES[config.scene_structure])
    
    # 3. SUBJECT
    if config.subject in SUBJECTS:
        prompt_parts.append(SUBJECTS[config.subject])
    
    # 4. ENVIRONMENT
    if config.environment in ENVIRONMENTS:
        prompt_parts.append(ENVIRONMENTS[config.environment])
    
    # 5. CAMERA / LENS (peut être multiple)
    for cam in config.camera:
        if cam in CAMERA_SETTINGS:
            prompt_parts.append(CAMERA_SETTINGS[cam])
    
    # 6. LIGHTING
    if config.lighting in LIGHTING_CONDITIONS:
        prompt_parts.append(LIGHTING_CONDITIONS[config.lighting])
    
    # 7. MATERIALS (peut être multiple)
    for mat in config.materials:
        if mat in MATERIALS:
            prompt_parts.append(MATERIALS[mat])
    
    # 8. STYLE (peut être multiple)
    for sty in config.style:
        if sty in STYLES:
            prompt_parts.append(STYLES[sty])
    
    # 9. CUSTOM POSITIVE
    prompt_parts.extend(config.custom_positive)
    
    # Assemblage final du prompt positif
    final_prompt = ", ".join(prompt_parts)
    
    # Construction du prompt négatif
    negative_parts = [get_full_negative_prompt()]
    negative_parts.extend(config.custom_negative)
    
    final_negative = ", ".join(negative_parts)
    
    return final_prompt, final_negative


def build_prompt_from_dict(
    user_prompt: str,
    scene_structure: str = "exterior",
    subject: str = "building",
    environment: str = "urban",
    camera: list[str] | str = None,
    lighting: str = "natural_daylight",
    materials: list[str] | str = None,
    style: list[str] | str = None,
    custom_positive: list[str] = None,
    custom_negative: list[str] = None
) -> tuple[str, str]:
    """
    Version simplifiée pour construire un prompt depuis des paramètres directs
    
    Args:
        user_prompt: Prompt de l'utilisateur
        scene_structure: Structure de scène (interior, exterior, aerial, etc.)
        subject: Sujet principal (building, facade, etc.)
        environment: Environnement (urban, residential, etc.)
        camera: Paramètres caméra (liste ou string)
        lighting: Conditions d'éclairage
        materials: Matériaux (liste ou string)
        style: Style photographique (liste ou string)
        custom_positive: Éléments positifs additionnels
        custom_negative: Éléments négatifs additionnels
    
    Returns:
        (prompt_positif, prompt_négatif)
    
    Example:
        >>> prompt, neg = build_prompt_from_dict(
        ...     user_prompt="modern villa with pool",
        ...     scene_structure="exterior",
        ...     subject="building",
        ...     environment="residential",
        ...     camera=["eye_level", "wide_angle", "straight_verticals"],
        ...     lighting="golden_hour",
        ...     materials=["concrete", "glass", "wood"],
        ...     style=["photorealistic", "architectural_photo", "high_quality"]
        ... )
    """
    
    config = PromptConfig()
    config.set_user_prompt(user_prompt)
    config.set_scene_structure(scene_structure)
    config.set_subject(subject)
    config.set_environment(environment)
    
    if camera:
        config.set_camera(camera)
    else:
        config.set_camera(["eye_level", "straight_verticals"])
    
    config.set_lighting(lighting)
    
    if materials:
        config.set_materials(materials)
    else:
        config.set_materials(["mixed_materials", "realistic_weathering"])
    
    if style:
        config.set_style(style)
    else:
        config.set_style(["photorealistic", "architectural_photo", "high_quality"])
    
    if custom_positive:
        for item in custom_positive:
            config.add_custom_positive(item)
    
    if custom_negative:
        for item in custom_negative:
            config.add_custom_negative(item)
    
    return build_modular_prompt(config)


def auto_detect_config_from_prompt(user_prompt: str) -> PromptConfig:
    """
    Détecte automatiquement les paramètres depuis le prompt utilisateur
    
    Cette fonction analyse le prompt utilisateur et suggère des configurations
    appropriées pour scene_structure, subject, environment, etc.
    
    Args:
        user_prompt: Prompt de l'utilisateur
    
    Returns:
        PromptConfig avec paramètres suggérés
    """
    
    config = PromptConfig()
    config.set_user_prompt(user_prompt)
    
    prompt_lower = user_prompt.lower()
    
    # Détection de la structure de scène
    if any(word in prompt_lower for word in ["interior", "room", "indoor", "inside"]):
        config.set_scene_structure("interior")
    elif any(word in prompt_lower for word in ["aerial", "top view", "bird eye", "overhead"]):
        config.set_scene_structure("aerial")
    elif any(word in prompt_lower for word in ["landscape", "site", "terrain", "masterplan"]):
        config.set_scene_structure("landscape")
    elif any(word in prompt_lower for word in ["detail", "close-up", "closeup", "zoom"]):
        config.set_scene_structure("detail")
    else:
        config.set_scene_structure("exterior")
    
    # Détection du sujet
    if any(word in prompt_lower for word in ["facade", "front", "elevation"]):
        config.set_subject("facade")
    elif any(word in prompt_lower for word in ["roof", "rooftop"]):
        config.set_subject("roof")
    elif any(word in prompt_lower for word in ["courtyard", "patio"]):
        config.set_subject("courtyard")
    elif any(word in prompt_lower for word in ["entrance", "entry", "door"]):
        config.set_subject("entrance")
    elif any(word in prompt_lower for word in ["urban", "block", "city"]):
        config.set_subject("urban_block")
    elif config.scene_structure == "interior":
        config.set_subject("interior_space")
    else:
        config.set_subject("building")
    
    # Détection de l'environnement
    if any(word in prompt_lower for word in ["park", "garden", "green"]):
        config.set_environment("park")
    elif any(word in prompt_lower for word in ["street", "road", "avenue"]):
        config.set_environment("street")
    elif any(word in prompt_lower for word in ["plaza", "square", "place"]):
        config.set_environment("plaza")
    elif any(word in prompt_lower for word in ["residential", "house", "villa", "apartment"]):
        config.set_environment("residential")
    elif any(word in prompt_lower for word in ["water", "river", "lake", "sea", "coast"]):
        config.set_environment("waterfront")
    else:
        config.set_environment("urban")
    
    # Détection de la caméra
    camera_settings = []
    
    if config.scene_structure == "aerial":
        if any(word in prompt_lower for word in ["orthogonal", "top-down", "plan view"]):
            camera_settings.append("aerial_orthogonal")
        else:
            camera_settings.append("aerial_oblique")
    else:
        if any(word in prompt_lower for word in ["low angle", "looking up"]):
            camera_settings.append("low_angle")
        elif any(word in prompt_lower for word in ["high angle", "looking down"]):
            camera_settings.append("high_angle")
        else:
            camera_settings.append("eye_level")
        
        if any(word in prompt_lower for word in ["wide", "wide angle"]):
            camera_settings.append("wide_angle")
        elif any(word in prompt_lower for word in ["telephoto", "zoom", "compressed"]):
            camera_settings.append("telephoto")
        else:
            camera_settings.append("normal_lens")
        
        camera_settings.append("straight_verticals")
    
    config.set_camera(camera_settings)
    
    # Détection de l'éclairage
    if any(word in prompt_lower for word in ["sunset", "golden hour", "warm light"]):
        config.set_lighting("golden_hour")
    elif any(word in prompt_lower for word in ["overcast", "cloudy", "soft light"]):
        config.set_lighting("overcast")
    elif any(word in prompt_lower for word in ["twilight", "dusk", "blue hour"]):
        config.set_lighting("blue_hour")
    elif any(word in prompt_lower for word in ["bright", "sunny", "sun"]):
        config.set_lighting("bright_sun")
    else:
        config.set_lighting("natural_daylight")
    
    # Détection des matériaux
    materials = []
    if any(word in prompt_lower for word in ["concrete", "béton"]):
        materials.append("concrete")
    if any(word in prompt_lower for word in ["brick", "brique"]):
        materials.append("brick")
    if any(word in prompt_lower for word in ["glass", "verre", "glazed"]):
        materials.append("glass")
    if any(word in prompt_lower for word in ["wood", "wooden", "timber", "bois"]):
        materials.append("wood")
    if any(word in prompt_lower for word in ["metal", "steel", "aluminum"]):
        materials.append("metal")
    if any(word in prompt_lower for word in ["stone", "pierre"]):
        materials.append("stone")
    
    if not materials:
        materials = ["mixed_materials"]
    
    materials.append("realistic_weathering")
    config.set_materials(materials)
    
    # Style par défaut
    config.set_style(["photorealistic", "architectural_photo", "high_quality", "natural_colors"])
    
    return config
