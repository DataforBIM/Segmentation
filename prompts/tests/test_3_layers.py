from intent.intent_parser import parse_intent
from intent.intent_config import IntentConfig
from prompts.modular_builder import auto_detect_config_from_prompt
from prompts.prompt_layer_builder import build_prompt_layers
import json

def test_3_layer_architecture():
    """Test de l'architecture à 3 couches"""
    
    test_prompts = [
        "Ajouter un peu de roses dans le jardin en premier plan",
        "Remplacer le ciel par un ciel étoilé",
        "Améliorer l'éclairage de la scène pour un effet dramatique"
    ]
    
    print("=" * 80)
    print("TEST ARCHITECTURE 3 COUCHES")
    print("=" * 80)
    
    for i, user_prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/3: {user_prompt}")
        print(f"{'='*80}\n")
        
        # Étape 1: Extraction d'intention
        print("📋 ÉTAPE 1: Extraction d'intention")
        intent_data = parse_intent(user_prompt)
        intent = IntentConfig(**intent_data)
        print(f"Intent: {json.dumps(intent_data, ensure_ascii=False, indent=2)}\n")
        
        # Étape 2: Auto-détection de configuration
        print("🔧 ÉTAPE 2: Auto-détection de configuration")
        prompt_config = auto_detect_config_from_prompt(user_prompt)
        print(f"Scene: {prompt_config.scene_structure}")
        print(f"Subject: {prompt_config.subject}")
        print(f"Environment: {prompt_config.environment}\n")
        
        # Étape 3: Construction des 3 couches
        print("🧱 ÉTAPE 3: Construction des 3 couches")
        layers = build_prompt_layers(prompt_config, intent)
        
        for layer in layers:
            print(f"\n🔹 LAYER {layer.role.upper()} (strength: {layer.strength})")
            print(f"   {layer.text[:150]}{'...' if len(layer.text) > 150 else ''}")
        
        # Étape 4: Rendu final
        print(f"\n{'='*80}")
        print("🎯 PROMPT FINAL ASSEMBLÉ")
        print(f"{'='*80}")
        final_prompt = ", ".join(layer.render() for layer in layers)
        print(f"{final_prompt}\n")
        
        print("-" * 80)
    
    print("\n✅ Tests terminés!")
    print("\n📊 RÉSUMÉ DE L'ARCHITECTURE:")
    print("   🔹 LAYER A (CORE):    Quoi + Où")
    print("   🔹 LAYER B (CONTEXT): Contraintes + Intégration")
    print("   🔹 LAYER C (QUALITY): Garde-fous visuels")

if __name__ == "__main__":
    test_3_layer_architecture()
