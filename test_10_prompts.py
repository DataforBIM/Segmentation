# Test avec 10 prompts variés - Prompt Layering
from prompts.builders import build_prompts
from prompts.modular_builder import auto_detect_config_from_prompt

print("=" * 100)
print("🎨 TEST DU SYSTÈME DE PROMPT LAYERING - 10 PROMPTS VARIÉS")
print("=" * 100)

# 10 prompts de test couvrant différents cas d'usage
test_prompts = [
    {
        "id": 1,
        "prompt": "change la façade en béton blanc moderne avec de grandes fenêtres en verre",
        "description": "Façade moderne - béton et verre"
    },
    {
        "id": 2,
        "prompt": "remplace le sol intérieur en marbre poli avec éclairage naturel",
        "description": "Sol intérieur luxueux"
    },
    {
        "id": 3,
        "prompt": "transforme le toit en tuiles rouges traditionnelles vue aérienne",
        "description": "Toiture vue du ciel"
    },
    {
        "id": 4,
        "prompt": "rénove l'entrée principale avec bois et métal dans un style contemporain",
        "description": "Entrée contemporaine"
    },
    {
        "id": 5,
        "prompt": "améliore la cour intérieure avec végétation et pierre naturelle",
        "description": "Courtyard paysagé"
    },
    {
        "id": 6,
        "prompt": "modernise le bloc urbain avec façades en brique et zinc en golden hour",
        "description": "Bloc urbain au coucher du soleil"
    },
    {
        "id": 7,
        "prompt": "détail de la façade en pierre avec texture vieillie et ombres douces",
        "description": "Gros plan matériaux"
    },
    {
        "id": 8,
        "prompt": "vue aérienne orthogonale du bâtiment en zone résidentielle avec toit vert",
        "description": "Plan masse résidentiel"
    },
    {
        "id": 9,
        "prompt": "espace intérieur minimaliste avec béton brut et lumière blue hour",
        "description": "Intérieur minimaliste crépuscule"
    },
    {
        "id": 10,
        "prompt": "immeuble en bord de mer avec grandes baies vitrées angle bas large",
        "description": "Architecture waterfront"
    }
]

for test in test_prompts:
    print(f"\n{'=' * 100}")
    print(f"TEST #{test['id']}: {test['description']}")
    print(f"{'=' * 100}")
    print(f"📝 Prompt utilisateur: \"{test['prompt']}\"")
    
    # Auto-détection de la configuration
    config = auto_detect_config_from_prompt(test['prompt'])
    
    print(f"\n🔍 COMPOSANTS AUTO-DÉTECTÉS:")
    print(f"   1️⃣  SCENE_STRUCTURE : {config.scene_structure}")
    print(f"   2️⃣  SUBJECT         : {config.subject}")
    print(f"   3️⃣  ENVIRONMENT     : {config.environment}")
    print(f"   4️⃣  CAMERA          : {config.camera}")
    print(f"   5️⃣  LIGHTING        : {config.lighting}")
    print(f"   6️⃣  MATERIALS       : {config.materials}")
    print(f"   7️⃣  STYLE           : {config.style}")
    
    # Construction du prompt final
    final_prompt, negative = build_prompts(
        user_prompt=test['prompt'],
        auto_detect=True
    )
    
    # Afficher le prompt utilisateur original
    print(f"\n📝 PROMPT UTILISATEUR ORIGINAL:")
    print(f"   ┌{'─' * 96}┐")
    print(f"   │ {test['prompt']:<94} │")
    print(f"   └{'─' * 96}┘")
    
    # Afficher le prompt final complet de manière structurée
    print(f"\n✨ PROMPT FINAL COMPLET GÉNÉRÉ:")
    print(f"   ┌{'─' * 96}┐")
    
    # Diviser en sections pour une meilleure lisibilité
    sections = final_prompt.split(", ")
    current_line = "   │ "
    for i, section in enumerate(sections):
        # Ajouter la section avec une virgule sauf pour la dernière
        text = section if i == len(sections) - 1 else section + ","
        
        # Si la ligne devient trop longue, on passe à la suivante
        if len(current_line) + len(text) + 1 > 96:
            # Remplir la ligne avec des espaces
            current_line += " " * (95 - len(current_line)) + "│"
            print(current_line)
            current_line = "   │ " + text + " "
        else:
            current_line += text + " "
    
    # Afficher la dernière ligne
    if len(current_line) > 5:
        current_line += " " * (95 - len(current_line)) + "│"
        print(current_line)
    
    print(f"   └{'─' * 96}┘")
    
    # Afficher le negative prompt complet
    print(f"\n🚫 NEGATIVE PROMPT COMPLET:")
    print(f"   ┌{'─' * 96}┐")
    
    neg_sections = negative.split(", ")
    current_line = "   │ "
    for i, section in enumerate(neg_sections):
        text = section if i == len(neg_sections) - 1 else section + ","
        
        if len(current_line) + len(text) + 1 > 96:
            current_line += " " * (95 - len(current_line)) + "│"
            print(current_line)
            current_line = "   │ " + text + " "
        else:
            current_line += text + " "
    
    if len(current_line) > 5:
        current_line += " " * (95 - len(current_line)) + "│"
        print(current_line)
    
    print(f"   └{'─' * 96}┘")
    
    # Statistiques
    print(f"\n📊 STATISTIQUES:")
    print(f"   • Longueur prompt positif : {len(final_prompt)} caractères")
    print(f"   • Longueur prompt négatif : {len(negative)} caractères")
    print(f"   • Total tokens (approx)   : {(len(final_prompt) + len(negative)) // 4}")
    print(f"   • Nombre de composants    : {len(final_prompt.split(', '))}")

print("\n" + "=" * 100)
print("✅ TEST TERMINÉ - Tous les prompts ont été analysés et construits")
print("=" * 100)

# Résumé des détections
print("\n📈 RÉSUMÉ DES DÉTECTIONS:\n")

summary_data = {
    "scene_structure": {},
    "subject": {},
    "environment": {},
    "lighting": {}
}

for test in test_prompts:
    config = auto_detect_config_from_prompt(test['prompt'])
    
    # Compter les occurrences
    summary_data["scene_structure"][config.scene_structure] = \
        summary_data["scene_structure"].get(config.scene_structure, 0) + 1
    summary_data["subject"][config.subject] = \
        summary_data["subject"].get(config.subject, 0) + 1
    summary_data["environment"][config.environment] = \
        summary_data["environment"].get(config.environment, 0) + 1
    summary_data["lighting"][config.lighting] = \
        summary_data["lighting"].get(config.lighting, 0) + 1

print("📐 SCENE STRUCTURES détectées:")
for key, count in summary_data["scene_structure"].items():
    print(f"   • {key:20} : {count} fois")

print("\n🎯 SUBJECTS détectés:")
for key, count in summary_data["subject"].items():
    print(f"   • {key:20} : {count} fois")

print("\n🌍 ENVIRONMENTS détectés:")
for key, count in summary_data["environment"].items():
    print(f"   • {key:20} : {count} fois")

print("\n💡 LIGHTING détecté:")
for key, count in summary_data["lighting"].items():
    print(f"   • {key:20} : {count} fois")

print("\n" + "=" * 100)
print("🎉 Analyse complète terminée!")
print("=" * 100)
