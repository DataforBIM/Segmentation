# Test du système de prompt layering (8 composants)
from prompts.builders import build_prompts
from prompts.modular_builder import auto_detect_config_from_prompt
from prompts.modular_structure import (
    SCENE_STRUCTURES, SUBJECTS, ENVIRONMENTS, CAMERA_SETTINGS, 
    LIGHTING_CONDITIONS, MATERIALS, STYLES
)
from PIL import Image
import requests
from io import BytesIO

print("=" * 80)
print("🎨 TEST DU PROMPT LAYERING (8 COMPOSANTS)")
print("=" * 80)

# Test 1: Auto-détection depuis prompt utilisateur
print("\n" + "=" * 80)
print("TEST 1: AUTO-DÉTECTION DEPUIS PROMPT UTILISATEUR")
print("=" * 80)

test_prompts = [
    "change la façade en blanc moderne",
    "remplace le sol en marbre",
    "transforme le toit en tuiles rouges",
    "change la couleur des volets en bleu",
]

for prompt in test_prompts:
    print(f"\n📝 Prompt utilisateur: \"{prompt}\"")
    
    config = auto_detect_config_from_prompt(prompt)
    
    print(f"\n   🔍 Composants détectés:")
    print(f"      • SCENE_STRUCTURE: {config.scene_structure}")
    print(f"      • SUBJECT: {config.subject}")
    print(f"      • ENVIRONMENT: {config.environment}")
    print(f"      • CAMERA: {config.camera}")
    print(f"      • LIGHTING: {config.lighting}")
    
    # Construire le prompt final
    final_prompt, negative = build_prompts(
        user_prompt=prompt,
        auto_detect=True
    )
    
    print(f"\n   ✨ PROMPT FINAL (tronqué à 200 chars):")
    print(f"      {final_prompt[:200]}...")
    print(f"\n   🚫 NEGATIVE (tronqué à 150 chars):")
    print(f"      {negative[:150]}...")

# Test 2: Configuration manuelle
print("\n" + "=" * 80)
print("TEST 2: CONFIGURATION MANUELLE DES COMPOSANTS")
print("=" * 80)

manual_configs = [
    {
        "name": "Façade moderne extérieure",
        "user_prompt": "change la façade en blanc",
        "scene_structure": "exterior",
        "subject": "building_facade",
        "environment": "urban",
        "camera": ["eye_level", "normal_lens"],
        "lighting": "natural_daylight",
        "materials": ["concrete", "glass"],
        "style": ["photorealistic", "architectural_photo"]
    },
    {
        "name": "Sol intérieur en marbre",
        "user_prompt": "remplace le sol en marbre",
        "scene_structure": "interior",
        "subject": "floor",
        "environment": "residential",
        "camera": ["eye_level", "wide_angle"],
        "lighting": "soft_interior",
        "materials": ["marble", "polished_stone"],
        "style": ["photorealistic", "interior_design"]
    },
    {
        "name": "Vue aérienne",
        "user_prompt": "améliore la vue du toit",
        "scene_structure": "aerial",
        "subject": "building_top",
        "environment": "urban",
        "camera": ["aerial_view", "drone_shot"],
        "lighting": "golden_hour",
        "materials": ["roof_tiles", "metal"],
        "style": ["photorealistic", "aerial_photo"]
    }
]

for config in manual_configs:
    print(f"\n🏗️  Configuration: {config['name']}")
    print(f"   📝 Prompt: {config['user_prompt']}")
    
    final_prompt, negative = build_prompts(
        user_prompt=config["user_prompt"],
        scene_structure=config.get("scene_structure"),
        subject=config.get("subject"),
        environment=config.get("environment"),
        camera=config.get("camera"),
        lighting=config.get("lighting"),
        materials=config.get("materials"),
        style=config.get("style"),
        auto_detect=False
    )
    
    print(f"\n   📊 Composants utilisés:")
    print(f"      • Structure: {config.get('scene_structure')}")
    print(f"      • Sujet: {config.get('subject')}")
    print(f"      • Environnement: {config.get('environment')}")
    print(f"      • Caméra: {config.get('camera')}")
    print(f"      • Éclairage: {config.get('lighting')}")
    print(f"      • Matériaux: {config.get('materials')}")
    print(f"      • Style: {config.get('style')}")
    
    print(f"\n   ✨ PROMPT FINAL:")
    # Diviser le prompt en lignes de 80 chars
    for i in range(0, len(final_prompt), 80):
        print(f"      {final_prompt[i:i+80]}")
    
    print(f"\n   🚫 NEGATIVE:")
    for i in range(0, len(negative), 80):
        print(f"      {negative[i:i+80]}")

# Test 3: Vérification de la structure des prompts
print("\n" + "=" * 80)
print("TEST 3: VÉRIFICATION DE LA STRUCTURE DES PROMPTS")
print("=" * 80)

print("\n📚 VOCABULAIRE DISPONIBLE PAR COMPOSANT:\n")

print("1️⃣  SCENE_STRUCTURE:")
for key in SCENE_STRUCTURES.keys():
    print(f"   • {key}")

print("\n2️⃣  SUBJECT (exemples):")
subjects = list(SUBJECTS.keys())[:10]
for key in subjects:
    print(f"   • {key}")
print(f"   ... et {len(SUBJECTS) - 10} autres")

print("\n3️⃣  ENVIRONMENT (exemples):")
environments = list(ENVIRONMENTS.keys())[:10]
for key in environments:
    print(f"   • {key}")
print(f"   ... et {len(ENVIRONMENTS) - 10} autres")

print("\n4️⃣  CAMERA (exemples):")
cameras = list(CAMERA_SETTINGS.keys())[:10]
for key in cameras:
    print(f"   • {key}")
print(f"   ... et {len(CAMERA_SETTINGS) - 10} autres")

print("\n5️⃣  LIGHTING:")
for key in LIGHTING_CONDITIONS.keys():
    print(f"   • {key}")

print("\n6️⃣  MATERIALS (exemples):")
materials = list(MATERIALS.keys())[:15]
for key in materials:
    print(f"   • {key}")
print(f"   ... et {len(MATERIALS) - 15} autres")

print("\n7️⃣  STYLE:")
for key in STYLES.keys():
    print(f"   • {key}")

print("\n8️⃣  NEGATIVE:")
print("   • Base artifacts")
print("   • Rendering issues")
print("   • Artistic styles")
print("   • Material issues")
print("   • Color issues")
print("   • Lighting problems")
print("   • Quality issues")
print("   • Geometry issues")

# Test 4: Test avec image réelle
print("\n" + "=" * 80)
print("TEST 4: TEST AVEC IMAGE RÉELLE")
print("=" * 80)

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image: {image.size}")

user_prompt = "change la façade en blanc moderne"
print(f"\n📝 Prompt: {user_prompt}")

# Auto-détection
final_prompt, negative = build_prompts(
    user_prompt=user_prompt,
    auto_detect=True
)

print(f"\n✨ PROMPT SDXL COMPLET (auto-détecté):")
print("=" * 80)
print(final_prompt)
print("\n🚫 NEGATIVE PROMPT:")
print("=" * 80)
print(negative)

# Test de longueur
print(f"\n📊 STATISTIQUES:")
print(f"   • Longueur prompt: {len(final_prompt)} caractères")
print(f"   • Longueur negative: {len(negative)} caractères")
print(f"   • Tokens estimés (prompt): ~{len(final_prompt.split())} mots")
print(f"   • Tokens estimés (negative): ~{len(negative.split())} mots")

# Vérifier la présence de tous les composants
print(f"\n🔍 VALIDATION DES COMPOSANTS:")
components_found = []
if any(s in final_prompt.lower() for s in ["exterior", "interior", "aerial"]):
    components_found.append("✅ SCENE_STRUCTURE")
else:
    components_found.append("❌ SCENE_STRUCTURE manquant")

if any(s in final_prompt.lower() for s in ["building", "facade", "wall", "floor"]):
    components_found.append("✅ SUBJECT")
else:
    components_found.append("❌ SUBJECT manquant")

if any(s in final_prompt.lower() for s in ["urban", "residential", "park"]):
    components_found.append("✅ ENVIRONMENT")
else:
    components_found.append("❌ ENVIRONMENT manquant")

if any(s in final_prompt.lower() for s in ["view", "lens", "angle", "perspective"]):
    components_found.append("✅ CAMERA")
else:
    components_found.append("❌ CAMERA manquant")

if any(s in final_prompt.lower() for s in ["daylight", "lighting", "golden hour"]):
    components_found.append("✅ LIGHTING")
else:
    components_found.append("❌ LIGHTING manquant")

if any(s in final_prompt.lower() for s in ["material", "concrete", "glass", "wood"]):
    components_found.append("✅ MATERIALS")
else:
    components_found.append("❌ MATERIALS manquant")

if any(s in final_prompt.lower() for s in ["photorealistic", "photo", "architectural"]):
    components_found.append("✅ STYLE")
else:
    components_found.append("❌ STYLE manquant")

if any(s in negative.lower() for s in ["artifacts", "noise", "blur"]):
    components_found.append("✅ NEGATIVE")
else:
    components_found.append("❌ NEGATIVE manquant")

for component in components_found:
    print(f"   {component}")

print("\n" + "=" * 80)
print("✅ TEST DU PROMPT LAYERING TERMINÉ")
print("=" * 80)

# Résumé
total_components = len(components_found)
valid_components = sum(1 for c in components_found if c.startswith("✅"))

print(f"\n📊 RÉSUMÉ:")
print(f"   • Composants validés: {valid_components}/{total_components}")
print(f"   • Taux de réussite: {valid_components/total_components*100:.1f}%")

if valid_components == total_components:
    print(f"\n   🎉 TOUS LES COMPOSANTS SONT PRÉSENTS!")
else:
    print(f"\n   ⚠️  Certains composants manquent dans le prompt généré")
