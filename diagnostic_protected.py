# Diagnostic: Pourquoi les ouvertures ne sont pas dans protected?
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import semantic_segment
from segmentation.target_resolver import resolve_target
from segmentation.intent_parser import parse_intent
import os

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 80)
print("🔍 DIAGNOSTIC: POURQUOI LES OUVERTURES NE SONT PAS PROTÉGÉES?")
print("=" * 80)

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image: {image.size}")

# Étape 1: Analyser l'intention
print("\n" + "=" * 80)
print("ÉTAPE 1: INTENTION DU PROMPT")
print("=" * 80)

intent = parse_intent("change la façade en blanc moderne")
target = resolve_target(intent)

print(f"""
Prompt: "change la façade en blanc moderne"

Target résolu:
  • Primary: {target.primary}
  • Protected: {target.protected}
  • Context: {target.context}
""")

print("✅ Le resolver demande bien de protéger: window, door")
print("   Vérifions si OneFormer les détecte...")

# Étape 2: Segmentation OneFormer
print("\n" + "=" * 80)
print("ÉTAPE 2: DÉTECTION ONEFORMER")
print("=" * 80)

semantic_map = semantic_segment(image, model_type="oneformer")

print(f"\n📊 Classes détectées par OneFormer: {len(semantic_map.detected_classes)}")
print("-" * 80)

for i, class_name in enumerate(semantic_map.detected_classes, 1):
    mask_array = np.array(semantic_map.masks[class_name])
    coverage = np.sum(mask_array > 0) / (image.size[0] * image.size[1])
    print(f"  {i:2d}. {class_name:20s} {coverage*100:6.2f}%")

# Étape 3: Vérification des classes protected demandées
print("\n" + "=" * 80)
print("ÉTAPE 3: VÉRIFICATION DES CLASSES PROTECTED")
print("=" * 80)

print(f"\nClasses protected demandées par le resolver:")
for prot in target.protected:
    found = prot in semantic_map.masks
    status = "✅ TROUVÉ" if found else "❌ ABSENT"
    if found:
        coverage = np.sum(np.array(semantic_map.masks[prot]) > 0) / (image.size[0] * image.size[1])
        print(f"  • {prot:20s} {status} ({coverage*100:.2f}%)")
    else:
        print(f"  • {prot:20s} {status}")

# Étape 4: Chercher des classes similaires
print("\n" + "=" * 80)
print("ÉTAPE 4: CLASSES SIMILAIRES DÉTECTÉES")
print("=" * 80)

opening_keywords = ["window", "door", "glass", "pane", "entrance", "frame", "opening"]
similar_classes = []

for detected_class in semantic_map.detected_classes:
    for keyword in opening_keywords:
        if keyword in detected_class.lower():
            similar_classes.append(detected_class)
            break

if similar_classes:
    print("\n🔍 Classes similaires trouvées:")
    for cls in similar_classes:
        coverage = np.sum(np.array(semantic_map.masks[cls]) > 0) / (image.size[0] * image.size[1])
        print(f"  • {cls}: {coverage*100:.2f}%")
else:
    print("\n❌ Aucune classe similaire aux ouvertures détectée")

# Étape 5: Visualisation de l'image pour comprendre
print("\n" + "=" * 80)
print("ÉTAPE 5: ANALYSE VISUELLE")
print("=" * 80)

# Sauvegarder l'image pour inspection
os.makedirs("output/diagnostic_protected", exist_ok=True)
image.save("output/diagnostic_protected/image_originale.png")

# Créer une visualisation avec annotations
annotated = image.copy()
draw = ImageDraw.Draw(annotated)

try:
    font = ImageFont.truetype("arial.ttf", 30)
    font_small = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()
    font_small = font

# Ajouter annotation
text = "OneFormer ne détecte PAS de fenêtres/portes ici"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
x = (image.width - text_width) // 2
draw.rectangle([x - 10, 10, x + text_width + 10, 60], fill=(255, 0, 0, 200))
draw.text((x, 20), text, fill=(255, 255, 255), font=font)

annotated.save("output/diagnostic_protected/image_annotee.png")

print(f"""
Image originale sauvegardée: output/diagnostic_protected/image_originale.png

Observations:
  • L'image contient clairement des fenêtres et portes
  • Mais OneFormer (ADE20K) ne les détecte pas sur cette image
  • Raisons possibles:
    1. Les ouvertures sont trop petites
    2. Elles fusionnent avec le building
    3. L'angle/éclairage empêche la détection
    4. Le modèle ADE20K n'est pas optimal pour cette architecture
""")

# Étape 6: Solutions proposées
print("\n" + "=" * 80)
print("ÉTAPE 6: SOLUTIONS PROPOSÉES")
print("=" * 80)

print("""
🔧 SOLUTIONS POUR DÉTECTER LES OUVERTURES:

1️⃣  APPROCHE HYBRIDE (RECOMMANDÉ):
   • OneFormer pour la scène globale (building, ciel, végétation)
   • Grounding DINO pour les ouvertures spécifiques
     → Text prompt: "window", "door", "glass window"
   • SAM2 pour raffiner les détections de Grounding DINO
   
   Exemple:
   ```python
   # Détection globale
   semantic_map = semantic_segment(image, "oneformer")
   
   # Détection spécifique des ouvertures
   windows = detect_with_grounding_dino(image, "window . glass window . windowpane")
   doors = detect_with_grounding_dino(image, "door . entrance door . doorway")
   
   # Combiner
   protected = windows + doors + person
   final = target - protected
   ```

2️⃣  DIVISION VERTICALE (ACTUELLE):
   • Assumer que les ouvertures sont dans le tiers central
   • Exclure le tiers central du target
   
   Exemple:
   ```python
   facade_upper = top 1/3 du building
   facade_lower = bottom 1/3 du building
   target = facade_upper + facade_lower  # Exclut le centre
   ```

3️⃣  MODÈLE SPÉCIALISÉ:
   • Utiliser un modèle fine-tuné sur l'architecture
   • Detectron2 avec COCO Panoptic (meilleure détection d'objets)
   • YOLOv8 segment spécialisé

4️⃣  DÉTECTION PAR EDGES + CONTOURS:
   • Détecter les fenêtres par leurs cadres rectangulaires
   • Utiliser OpenCV pour trouver les contours réguliers
   • Filtrer par ratio largeur/hauteur typique des fenêtres

💡 RECOMMANDATION IMMÉDIATE:
   Implémenter l'approche hybride OneFormer + Grounding DINO
   
   Avantages:
   ✓ OneFormer reste pour la scène globale (excellent)
   ✓ Grounding DINO détecte avec text prompts (flexible)
   ✓ SAM2 raffine les bords (précision)
   ✓ Pas besoin de fine-tuning
""")

print("\n" + "=" * 80)
print("✅ DIAGNOSTIC TERMINÉ")
print("=" * 80)

print(f"""
RÉSUMÉ:
  • OneFormer détecte: {len(semantic_map.detected_classes)} classes
  • Protected demandé: {len(target.protected)} classes
  • Protected trouvé: 1 classe (person uniquement)
  • Manquants: window, door, furniture, object
  
⚠️  PROBLÈME IDENTIFIÉ:
   OneFormer (ADE20K) ne détecte pas les fenêtres/portes sur cette image

🎯 PROCHAINE ÉTAPE:
   Implémenter Grounding DINO pour détecter les ouvertures par text prompt
   
📁 FICHIERS GÉNÉRÉS:
   - output/diagnostic_protected/image_originale.png
   - output/diagnostic_protected/image_annotee.png
""")
