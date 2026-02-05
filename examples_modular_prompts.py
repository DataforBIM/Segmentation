"""
Exemples d'utilisation du nouveau système de prompts modulaires
"""

from pipeline import run_pipeline

# =============================================================================
# EXEMPLE 1: Mode Auto-Détection (Simple et Rapide)
# =============================================================================

def example_auto_detection():
    """Le système détecte automatiquement les paramètres depuis votre prompt"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="modern concrete building with glass facade in urban area",
        
        # Mode auto-détection activé (par défaut)
        auto_detect_prompt=True,
        
        # Étapes du pipeline
        enable_controlnet=True,
        enable_sdxl=True,
        enable_refiner=False,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Image générée avec auto-détection")
    return result


# =============================================================================
# EXEMPLE 2: Mode Manuel (Contrôle Total)
# =============================================================================

def example_manual_configuration():
    """Configuration manuelle complète de tous les paramètres"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="villa de luxe avec piscine",
        
        # Configuration manuelle du prompt
        scene_structure="exterior",
        subject="building",
        environment="residential",
        camera=["eye_level", "wide_angle", "straight_verticals"],
        lighting="golden_hour",
        materials=["concrete", "glass", "wood"],
        style=["photorealistic", "architectural_photo", "high_quality", "natural_colors"],
        
        # Désactiver l'auto-détection
        auto_detect_prompt=False,
        
        # Étapes du pipeline
        enable_controlnet=True,
        enable_sdxl=True,
        enable_refiner=True,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Image générée avec configuration manuelle")
    return result


# =============================================================================
# EXEMPLE 3: Mode Hybride (Auto + Overrides)
# =============================================================================

def example_hybrid_mode():
    """Combine auto-détection avec quelques overrides manuels"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="building renovation project",
        
        # Auto-détection activée
        auto_detect_prompt=True,
        
        # Mais avec quelques overrides spécifiques
        lighting="golden_hour",  # Forcer le golden hour
        materials=["brick", "wood", "realistic_weathering"],  # Forcer ces matériaux
        
        # Étapes du pipeline
        enable_controlnet=True,
        enable_sdxl=True,
        enable_refiner=False,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Image générée en mode hybride")
    return result


# =============================================================================
# EXEMPLE 4: Scène Intérieure
# =============================================================================

def example_interior_scene():
    """Génération d'un espace intérieur"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="modern living room with large windows and natural light",
        
        scene_structure="interior",
        subject="interior_space",
        environment="residential",
        camera=["eye_level", "wide_angle"],
        lighting="natural_daylight",
        materials=["wood", "concrete", "glass"],
        style=["photorealistic", "architectural_photo", "natural_colors"],
        
        auto_detect_prompt=False,
        enable_controlnet=True,
        enable_sdxl=True,
        enable_refiner=False,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Scène intérieure générée")
    return result


# =============================================================================
# EXEMPLE 5: Vue Aérienne
# =============================================================================

def example_aerial_view():
    """Génération d'une vue aérienne urbaine"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="urban block aerial reconstruction, correct geometry",
        
        scene_structure="aerial",
        subject="urban_block",
        environment="urban",
        camera=["aerial_oblique"],
        lighting="overcast",
        materials=["mixed_materials", "realistic_weathering"],
        style=["photorealistic", "documentary", "high_quality"],
        
        auto_detect_prompt=False,
        enable_controlnet=True,
        enable_segmentation=True,
        enable_sdxl=True,
        enable_refiner=False,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Vue aérienne générée")
    return result


# =============================================================================
# EXEMPLE 6: Façade Architecture
# =============================================================================

def example_facade():
    """Génération d'une façade architecturale"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="brick facade with wooden windows and door",
        
        scene_structure="exterior",
        subject="facade",
        environment="residential",
        camera=["eye_level", "normal_lens", "straight_verticals"],
        lighting="natural_daylight",
        materials=["brick", "wood", "realistic_weathering"],
        style=["photorealistic", "architectural_photo", "documentary"],
        
        auto_detect_prompt=False,
        enable_controlnet=True,
        enable_segmentation=True,
        enable_sdxl=True,
        enable_refiner=False,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Façade générée")
    return result


# =============================================================================
# EXEMPLE 7: Détail Architectural
# =============================================================================

def example_architectural_detail():
    """Génération d'un détail architectural en gros plan"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="architectural detail, entrance design",
        
        scene_structure="detail",
        subject="entrance",
        environment="isolated",
        camera=["eye_level", "normal_lens"],
        lighting="soft_shadows",
        materials=["metal", "glass", "concrete"],
        style=["photorealistic", "high_quality", "clean_composition"],
        
        auto_detect_prompt=False,
        enable_controlnet=True,
        enable_sdxl=True,
        enable_refiner=True,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Détail architectural généré")
    return result


# =============================================================================
# EXEMPLE 8: Paysage Architectural
# =============================================================================

def example_landscape():
    """Génération d'un paysage architectural"""
    
    result = run_pipeline(
        image_url="https://res.cloudinary.com/...",
        user_prompt="architectural landscape with building in park context",
        
        scene_structure="landscape",
        subject="building",
        environment="park",
        camera=["eye_level", "wide_angle"],
        lighting="golden_hour",
        materials=["concrete", "glass", "wood"],
        style=["photorealistic", "architectural_photo", "natural_colors"],
        
        auto_detect_prompt=False,
        enable_controlnet=True,
        enable_sdxl=True,
        enable_refiner=False,
        enable_upscaler=False,
        enable_upload=False
    )
    
    print("✅ Paysage architectural généré")
    return result


# =============================================================================
# TEST: Construire un Prompt Directement
# =============================================================================

def test_prompt_builder():
    """Test du builder de prompts sans exécuter le pipeline"""
    
    from prompts.modular_builder import build_prompt_from_dict
    
    prompt, negative = build_prompt_from_dict(
        user_prompt="modern villa with infinity pool",
        scene_structure="exterior",
        subject="building",
        environment="residential",
        camera=["eye_level", "wide_angle", "straight_verticals"],
        lighting="golden_hour",
        materials=["concrete", "glass", "wood"],
        style=["photorealistic", "architectural_photo", "high_quality"],
        custom_positive=["luxury design", "infinity pool"],
        custom_negative=["old", "deteriorated", "abandoned"]
    )
    
    print("\n" + "="*80)
    print("PROMPT POSITIF:")
    print("="*80)
    print(prompt)
    print("\n" + "="*80)
    print("PROMPT NÉGATIF:")
    print("="*80)
    print(negative)
    print("="*80 + "\n")


# =============================================================================
# TEST: Auto-Détection
# =============================================================================

def test_auto_detection():
    """Test de l'auto-détection des paramètres"""
    
    from prompts.modular_builder import auto_detect_config_from_prompt, build_modular_prompt
    
    test_prompts = [
        "modern concrete building with large glass windows in urban area at sunset",
        "interior living room with wooden floor and natural light",
        "aerial view of urban block with corrected geometry",
        "brick facade with wooden windows",
        "architectural detail of entrance with metal door"
    ]
    
    for test_prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"TEST: {test_prompt}")
        print('='*80)
        
        config = auto_detect_config_from_prompt(test_prompt)
        prompt, negative = build_modular_prompt(config)
        
        print(f"\nCONFIG DÉTECTÉE:")
        print(f"  - scene_structure: {config.scene_structure}")
        print(f"  - subject: {config.subject}")
        print(f"  - environment: {config.environment}")
        print(f"  - camera: {config.camera}")
        print(f"  - lighting: {config.lighting}")
        print(f"  - materials: {config.materials}")
        
        print(f"\nPROMPT GÉNÉRÉ: {prompt[:150]}...")


if __name__ == "__main__":
    print("🚀 Tests du système de prompts modulaires\n")
    
    # Tester le builder de prompts
    print("📝 Test 1: Construction de prompt...")
    test_prompt_builder()
    
    # Tester l'auto-détection
    print("\n🧠 Test 2: Auto-détection...")
    test_auto_detection()
    
    print("\n✅ Tests terminés!")
    print("\nPour exécuter les exemples de pipeline, décommentez les fonctions ci-dessus.")
