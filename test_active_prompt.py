# Test: Afficher le prompt actif pour l'image input
from PIL import Image
import requests
from io import BytesIO
from prompts.builders import build_prompts

# Image de test
IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 80)
print("🔍 PROMPT ACTIF POUR L'IMAGE INPUT")
print("=" * 80)

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image: {image.size}")
print(f"   📍 URL: {IMAGE_URL}")

# Prompt utilisateur (comme dans run.py)
user_prompt = "Change la couleur de la façade"

print(f"\n📝 Prompt utilisateur: \"{user_prompt}\"")

# Configuration exacte du pipeline (comme dans run.py)
prompt_config = {
    "user_prompt": user_prompt,
    "scene_structure": None,         # auto-détection
    "subject": None,                 # auto-détection
    "environment": None,
    "camera": None,
    "lighting": None,
    "materials": None,
    "style": None,
    "auto_detect": True              # ✅ Auto-détection activée
}

print("\n🧠 Configuration:")
print("   • Mode: Auto-détection")
print("   • Scene structure: auto")
print("   • Subject: auto")
print("   • Environment: auto")

# Générer le prompt (exactement comme le pipeline)
print("\n⚙️  Génération du prompt...")
final_prompt, negative_prompt = build_prompts(**prompt_config)

print("\n" + "=" * 80)
print("✨ PROMPT POSITIF COMPLET")
print("=" * 80)
print(final_prompt)

print("\n" + "=" * 80)
print("🚫 PROMPT NÉGATIF COMPLET")
print("=" * 80)
print(negative_prompt)

# Statistiques
print("\n" + "=" * 80)
print("📊 STATISTIQUES")
print("=" * 80)
print(f"   • Longueur prompt positif: {len(final_prompt)} caractères")
print(f"   • Longueur prompt négatif: {len(negative_prompt)} caractères")
print(f"   • Mots prompt positif: ~{len(final_prompt.split())} mots")
print(f"   • Mots prompt négatif: ~{len(negative_prompt.split())} mots")

# Décomposition par composants
print("\n" + "=" * 80)
print("🔍 ANALYSE PAR COMPOSANTS")
print("=" * 80)

components = {
    "User prompt": user_prompt,
    "Scene structure": None,
    "Subject": None,
    "Environment": None,
    "Camera": None,
    "Lighting": None,
    "Materials": None,
    "Style": None,
}

# Détecter les composants dans le prompt
if "exterior" in final_prompt.lower():
    components["Scene structure"] = "exterior"
elif "interior" in final_prompt.lower():
    components["Scene structure"] = "interior"
elif "aerial" in final_prompt.lower():
    components["Scene structure"] = "aerial"

if "building" in final_prompt.lower():
    components["Subject"] = "building/facade"
elif "floor" in final_prompt.lower():
    components["Subject"] = "floor"
elif "wall" in final_prompt.lower():
    components["Subject"] = "wall"

if "urban" in final_prompt.lower():
    components["Environment"] = "urban"
elif "residential" in final_prompt.lower():
    components["Environment"] = "residential"

if "eye level" in final_prompt.lower():
    components["Camera"] = "eye_level + normal_lens"
elif "wide angle" in final_prompt.lower():
    components["Camera"] = "wide_angle"

if "daylight" in final_prompt.lower():
    components["Lighting"] = "natural_daylight"
elif "golden hour" in final_prompt.lower():
    components["Lighting"] = "golden_hour"

if "material" in final_prompt.lower():
    components["Materials"] = "mixed_materials + weathering"

if "photorealistic" in final_prompt.lower():
    components["Style"] = "photorealistic + architectural_photo"

print("\n1️⃣  USER PROMPT:")
print(f"   → {components['User prompt']}")

print("\n2️⃣  SCENE STRUCTURE:")
print(f"   → {components['Scene structure'] or 'non détecté'}")

print("\n3️⃣  SUBJECT:")
print(f"   → {components['Subject'] or 'non détecté'}")

print("\n4️⃣  ENVIRONMENT:")
print(f"   → {components['Environment'] or 'non détecté'}")

print("\n5️⃣  CAMERA:")
print(f"   → {components['Camera'] or 'non détecté'}")

print("\n6️⃣  LIGHTING:")
print(f"   → {components['Lighting'] or 'non détecté'}")

print("\n7️⃣  MATERIALS:")
print(f"   → {components['Materials'] or 'non détecté'}")

print("\n8️⃣  STYLE:")
print(f"   → {components['Style'] or 'non détecté'}")

# Extrait du prompt par section
print("\n" + "=" * 80)
print("📝 APERÇU DU PROMPT PAR SECTION")
print("=" * 80)

sections = final_prompt.split(", ")
print(f"\nNombre de sections: {len(sections)}")
print("\nPremières 10 sections:")
for i, section in enumerate(sections[:10], 1):
    print(f"   {i}. {section}")

print("\n...")
print(f"\nDernières 5 sections:")
for i, section in enumerate(sections[-5:], len(sections)-4):
    print(f"   {i}. {section}")

# Mots-clés importants
print("\n" + "=" * 80)
print("🎯 MOTS-CLÉS IMPORTANTS DÉTECTÉS")
print("=" * 80)

keywords = {
    "Architecture": ["building", "facade", "exterior", "interior", "architectural"],
    "Qualité": ["photorealistic", "8k", "high definition", "professional"],
    "Perspective": ["eye level", "perspective", "view"],
    "Éclairage": ["daylight", "natural", "lighting"],
    "Matériaux": ["material", "concrete", "glass", "weathering"],
    "Style": ["photorealistic", "architectural photo", "raw photograph"],
}

for category, words in keywords.items():
    found = [w for w in words if w in final_prompt.lower()]
    if found:
        print(f"\n{category}:")
        for word in found:
            print(f"   ✅ {word}")

print("\n" + "=" * 80)
print("✅ ANALYSE TERMINÉE")
print("=" * 80)

print("""
💡 CE PROMPT SERA ENVOYÉ À SDXL:
   Ce prompt contient tous les composants nécessaires pour générer
   une image architecturale de haute qualité avec:
   - Structure de scène appropriée
   - Sujet clairement défini
   - Environnement contextualisé
   - Paramètres de caméra professionnels
   - Éclairage naturel
   - Matériaux réalistes
   - Style photographique authentique
   - Négatif prompt pour éviter les artefacts

📌 POUR MODIFIER LE PROMPT:
   Éditez run.py et modifiez les paramètres:
   - scene_structure="exterior"  # ou "interior", "aerial"
   - subject="building_facade"   # ou autre
   - environment="urban"         # ou autre
   - etc.
""")
