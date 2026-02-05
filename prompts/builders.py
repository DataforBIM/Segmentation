# =====================================================
# PROMPT BUILDER - MODULAR APPROACH
# prompts/builders.py
# =====================================================
# Utilise la nouvelle structure modulaire au lieu de la détection de scène

from prompts.modular_builder import (
    build_prompt_from_dict,
    auto_detect_config_from_prompt,
    PromptConfig
)

def build_prompts(
    user_prompt: str,
    scene_structure: str = None,
    subject: str = None,
    environment: str = None,
    camera: list[str] | str = None,
    lighting: str = None,
    materials: list[str] | str = None,
    style: list[str] | str = None,
    custom_positive: list[str] = None,
    custom_negative: list[str] = None,
    auto_detect: bool = True
) -> tuple[str, str]:
    """
    Construit un prompt en utilisant la structure modulaire
    
    Args:
        user_prompt: Prompt de l'utilisateur (obligatoire)
        scene_structure: Structure de scène (interior, exterior, aerial, landscape, detail)
        subject: Sujet principal (building, facade, interior_space, etc.)
        environment: Environnement (urban, residential, park, etc.)
        camera: Paramètres caméra (eye_level, wide_angle, etc.)
        lighting: Conditions d'éclairage (natural_daylight, golden_hour, etc.)
        materials: Matériaux (concrete, glass, wood, etc.)
        style: Style photographique (photorealistic, architectural_photo, etc.)
        custom_positive: Éléments positifs additionnels
        custom_negative: Éléments négatifs additionnels
        auto_detect: Si True, détecte automatiquement les paramètres depuis le prompt
    
    Returns:
        (prompt_positif, prompt_négatif)
    
    Example avec auto-détection:
        >>> prompt, neg = build_prompts("modern concrete building in urban area")
    
    Example avec paramètres manuels:
        >>> prompt, neg = build_prompts(
        ...     user_prompt="villa with pool",
        ...     scene_structure="exterior",
        ...     subject="building",
        ...     environment="residential",
        ...     camera=["eye_level", "wide_angle"],
        ...     lighting="golden_hour",
        ...     materials=["concrete", "glass"],
        ...     auto_detect=False
        ... )
    """
    
    # MODE AUTO: Détection automatique depuis le prompt
    if auto_detect and not scene_structure:
        config = auto_detect_config_from_prompt(user_prompt)
        
        # Appliquer les overrides si fournis
        if subject:
            config.set_subject(subject)
        if environment:
            config.set_environment(environment)
        if camera:
            config.set_camera(camera)
        if lighting:
            config.set_lighting(lighting)
        if materials:
            config.set_materials(materials)
        if style:
            config.set_style(style)
        if custom_positive:
            for item in custom_positive:
                config.add_custom_positive(item)
        if custom_negative:
            for item in custom_negative:
                config.add_custom_negative(item)
        
        from prompts.modular_builder import build_modular_prompt
        return build_modular_prompt(config)
    
    # MODE MANUEL: Utilisation des paramètres fournis
    return build_prompt_from_dict(
        user_prompt=user_prompt,
        scene_structure=scene_structure or "exterior",
        subject=subject or "building",
        environment=environment or "urban",
        camera=camera,
        lighting=lighting or "natural_daylight",
        materials=materials,
        style=style,
        custom_positive=custom_positive,
        custom_negative=custom_negative
    )


# =====================================================
# BACKWARD COMPATIBILITY - Ancienne interface
# =====================================================
# Pour maintenir la compatibilité avec l'ancien code qui utilise scene_type

def build_prompts_legacy(
    scene_type: str,
    user_prompt: str,
    aerial_elements: list[str] = None
) -> tuple[str, str]:
    """
    Interface de compatibilité avec l'ancien système basé sur scene_type
    
    Args:
        scene_type: Type de scène (INTERIOR, EXTERIOR, AERIAL)
        user_prompt: Prompt de l'utilisateur
        aerial_elements: Éléments aériens (ignoré dans la nouvelle version)
    
    Returns:
        (prompt_positif, prompt_négatif)
    """
    
    # Mapping de l'ancien système vers le nouveau
    scene_mapping = {
        "INTERIOR": "interior",
        "EXTERIOR": "exterior",
        "AERIAL": "aerial"
    }
    
    scene_structure = scene_mapping.get(scene_type, "exterior")
    
    # Utiliser la nouvelle logique modulaire
    return build_prompts(
        user_prompt=user_prompt,
        scene_structure=scene_structure,
        auto_detect=False  # Pas d'auto-détection en mode legacy
    )
