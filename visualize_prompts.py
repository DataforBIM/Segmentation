"""
Visualisation de la structure de prompts modulaires
Exécutez ce fichier pour voir comment les prompts sont construits
"""

from prompts.modular_builder import build_prompt_from_dict, auto_detect_config_from_prompt, build_modular_prompt


def visualize_prompt_structure():
    """Affiche la structure détaillée d'un prompt modulaire"""
    
    print("\n" + "="*100)
    print(" "*35 + "STRUCTURE DE PROMPT MODULAIRE")
    print("="*100 + "\n")
    
    # Configuration exemple
    config = {
        "user_prompt": "modern concrete villa with large glass windows",
        "scene_structure": "exterior",
        "subject": "building",
        "environment": "residential",
        "camera": ["eye_level", "wide_angle", "straight_verticals"],
        "lighting": "golden_hour",
        "materials": ["concrete", "glass", "wood"],
        "style": ["photorealistic", "architectural_photo", "high_quality"]
    }
    
    prompt, negative = build_prompt_from_dict(**config)
    
    # Afficher la configuration
    print("📋 CONFIGURATION:")
    print("-" * 100)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    
    # Afficher le prompt positif
    print("\n" + "="*100)
    print("✅ PROMPT POSITIF FINAL:")
    print("="*100)
    print(f"\n{prompt}\n")
    
    # Découper le prompt en sections
    print("="*100)
    print("📦 SECTIONS DU PROMPT:")
    print("="*100)
    
    sections = prompt.split(", ")
    current_section = "USER PROMPT"
    section_content = []
    
    print(f"\n[{current_section}]")
    for i, part in enumerate(sections):
        if i < 5:  # Approximation des sections
            print(f"  • {part}")
    
    # Afficher le prompt négatif
    print("\n" + "="*100)
    print("❌ PROMPT NÉGATIF FINAL:")
    print("="*100)
    print(f"\n{negative}\n")
    
    print("="*100)
    print(f"📊 STATS:")
    print("-" * 100)
    print(f"  Longueur prompt positif:  {len(prompt)} caractères")
    print(f"  Longueur prompt négatif:  {len(negative)} caractères")
    print(f"  Nombre de tokens estimés: ~{(len(prompt) + len(negative)) // 4}")
    print("="*100 + "\n")


def compare_prompts():
    """Compare différentes configurations de prompts"""
    
    print("\n" + "="*100)
    print(" "*35 + "COMPARAISON DE PROMPTS")
    print("="*100 + "\n")
    
    test_configs = [
        {
            "name": "Villa Moderne",
            "user_prompt": "modern villa with pool",
            "scene_structure": "exterior",
            "subject": "building",
            "environment": "residential",
            "lighting": "golden_hour",
            "materials": ["concrete", "glass"]
        },
        {
            "name": "Intérieur Contemporain",
            "user_prompt": "contemporary living room",
            "scene_structure": "interior",
            "subject": "interior_space",
            "environment": "residential",
            "lighting": "natural_daylight",
            "materials": ["wood", "concrete"]
        },
        {
            "name": "Vue Aérienne Urbaine",
            "user_prompt": "urban block aerial view",
            "scene_structure": "aerial",
            "subject": "urban_block",
            "environment": "urban",
            "lighting": "overcast",
            "materials": ["mixed_materials"]
        }
    ]
    
    for config in test_configs:
        name = config.pop("name")
        prompt, _ = build_prompt_from_dict(**config)
        
        print(f"\n{'─'*100}")
        print(f"🏗️  {name}")
        print(f"{'─'*100}")
        print(f"Config: {config['user_prompt']}")
        print(f"       scene={config['scene_structure']}, lighting={config['lighting']}")
        print(f"\nPrompt: {prompt[:150]}...")
        print(f"        ({len(prompt)} caractères)")


def test_auto_detection_visual():
    """Visualise l'auto-détection des paramètres"""
    
    print("\n" + "="*100)
    print(" "*30 + "AUTO-DÉTECTION DE PARAMÈTRES")
    print("="*100 + "\n")
    
    test_prompts = [
        "modern concrete building with large glass windows in urban area at sunset",
        "cozy interior living room with wooden floor and natural light",
        "aerial view of residential urban block with corrected geometry",
        "brick facade with wooden windows and metal door",
    ]
    
    for test_prompt in test_prompts:
        print(f"\n{'─'*100}")
        print(f"📝 PROMPT: {test_prompt}")
        print(f"{'─'*100}")
        
        config = auto_detect_config_from_prompt(test_prompt)
        
        print(f"\n🧠 DÉTECTION AUTOMATIQUE:")
        print(f"   scene_structure : {config.scene_structure}")
        print(f"   subject         : {config.subject}")
        print(f"   environment     : {config.environment}")
        print(f"   camera          : {', '.join(config.camera)}")
        print(f"   lighting        : {config.lighting}")
        print(f"   materials       : {', '.join(config.materials)}")
        
        prompt, _ = build_modular_prompt(config)
        print(f"\n📤 PROMPT GÉNÉRÉ: {prompt[:120]}...")


def show_all_modules():
    """Affiche tous les modules disponibles"""
    
    print("\n" + "="*100)
    print(" "*30 + "MODULES DISPONIBLES")
    print("="*100 + "\n")
    
    from prompts.modular_structure import (
        SCENE_STRUCTURES, SUBJECTS, ENVIRONMENTS,
        CAMERA_SETTINGS, LIGHTING_CONDITIONS, MATERIALS, STYLES
    )
    
    modules = [
        ("SCENE STRUCTURES", SCENE_STRUCTURES),
        ("SUBJECTS", SUBJECTS),
        ("ENVIRONMENTS", ENVIRONMENTS),
        ("CAMERA SETTINGS", CAMERA_SETTINGS),
        ("LIGHTING CONDITIONS", LIGHTING_CONDITIONS),
        ("MATERIALS", MATERIALS),
        ("STYLES", STYLES)
    ]
    
    for module_name, module_dict in modules:
        print(f"\n{'─'*100}")
        print(f"📦 {module_name}")
        print(f"{'─'*100}")
        
        for key in sorted(module_dict.keys()):
            value = module_dict[key]
            # Afficher seulement le début si trop long
            display_value = value[:80] + "..." if len(value) > 80 else value
            print(f"  • {key:25s} : {display_value}")
    
    print("\n" + "="*100 + "\n")


def interactive_builder():
    """Interface interactive pour construire un prompt"""
    
    print("\n" + "="*100)
    print(" "*30 + "CONSTRUCTEUR INTERACTIF")
    print("="*100 + "\n")
    
    print("Entrez vos paramètres (appuyez sur Entrée pour utiliser la valeur par défaut)")
    print("-" * 100)
    
    user_prompt = input("\n📝 User Prompt: ") or "modern building"
    scene_structure = input("🏗️  Scene Structure (exterior/interior/aerial): ") or "exterior"
    subject = input("🎯 Subject (building/facade/interior_space): ") or "building"
    environment = input("🌍 Environment (urban/residential/park): ") or "urban"
    lighting = input("💡 Lighting (natural_daylight/golden_hour/overcast): ") or "natural_daylight"
    
    materials_input = input("🧱 Materials (séparés par des virgules): ") or "concrete,glass"
    materials = [m.strip() for m in materials_input.split(",")]
    
    camera_input = input("📷 Camera (séparés par des virgules): ") or "eye_level,wide_angle"
    camera = [c.strip() for c in camera_input.split(",")]
    
    style_input = input("🎨 Style (séparés par des virgules): ") or "photorealistic,high_quality"
    style = [s.strip() for s in style_input.split(",")]
    
    print("\n" + "="*100)
    print("⚙️  CONSTRUCTION DU PROMPT...")
    print("="*100)
    
    prompt, negative = build_prompt_from_dict(
        user_prompt=user_prompt,
        scene_structure=scene_structure,
        subject=subject,
        environment=environment,
        camera=camera,
        lighting=lighting,
        materials=materials,
        style=style
    )
    
    print("\n✅ PROMPT POSITIF:")
    print("-" * 100)
    print(prompt)
    
    print("\n❌ PROMPT NÉGATIF:")
    print("-" * 100)
    print(negative)
    
    print("\n" + "="*100)
    print(f"📊 Longueur: {len(prompt)} caractères (positif) + {len(negative)} caractères (négatif)")
    print("="*100 + "\n")


def menu():
    """Menu principal"""
    
    print("\n" + "="*100)
    print(" "*25 + "VISUALISATION DE PROMPTS MODULAIRES")
    print("="*100)
    print("\n  1. Visualiser la structure d'un prompt")
    print("  2. Comparer différents prompts")
    print("  3. Tester l'auto-détection")
    print("  4. Afficher tous les modules disponibles")
    print("  5. Constructeur interactif")
    print("  0. Quitter")
    print("\n" + "="*100)
    
    choice = input("\nVotre choix: ")
    
    if choice == "1":
        visualize_prompt_structure()
    elif choice == "2":
        compare_prompts()
    elif choice == "3":
        test_auto_detection_visual()
    elif choice == "4":
        show_all_modules()
    elif choice == "5":
        interactive_builder()
    elif choice == "0":
        print("\n👋 Au revoir!\n")
        return False
    else:
        print("\n❌ Choix invalide\n")
    
    return True


if __name__ == "__main__":
    # Mode automatique: exécute toutes les visualisations
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("\n🚀 MODE AUTOMATIQUE - EXÉCUTION DE TOUTES LES VISUALISATIONS\n")
        visualize_prompt_structure()
        compare_prompts()
        test_auto_detection_visual()
        show_all_modules()
        print("\n✅ Toutes les visualisations terminées!\n")
    else:
        # Mode interactif
        while menu():
            pass
