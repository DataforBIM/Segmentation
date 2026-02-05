# Test de raffinement morphologique (plus adapté pour les façades)
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import semantic_segment
from segmentation.mask_refinement import refine_mask
import os

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 60)
print("🎯 TEST RAFFINEMENT MORPHOLOGIQUE")
print("=" * 60)
print("""
Pour les façades, le raffinement morphologique est plus adapté
que SAM2 car:

1. SAM2 détecte des OBJETS isolés (chaise, table, personne)
2. Une façade est une SCÈNE avec plusieurs éléments
3. Les opérations morphologiques (dilate/erode/feather) 
   nettoient les bords sans réduire la région

PIPELINE:
  PASSE 1: OneFormer (détection sémantique)
  PASSE 2: Division verticale (upper/lower)
  PASSE 3: Raffinement morphologique (clean edges)
""")

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image: {image.size}")

# PASSE 1: Segmentation sémantique
print("\n" + "=" * 60)
print("PASSE 1: SEGMENTATION SÉMANTIQUE (OneFormer)")
print("=" * 60)

semantic_map = semantic_segment(image, model_type="oneformer")
building_mask = semantic_map.masks.get("building")
if not building_mask:
    print("❌ Aucune classe 'building' détectée")
    exit()

building_coverage = np.sum(np.array(building_mask) > 127) / (image.size[0] * image.size[1])
print(f"✅ Building détecté: {building_coverage*100:.1f}%")

# PASSE 2: Division verticale
print("\n" + "=" * 60)
print("PASSE 2: DIVISION VERTICALE")
print("=" * 60)

mask_array = np.array(building_mask)
height, width = mask_array.shape

# Trouver les limites
rows_with_building = np.any(mask_array > 0, axis=1)
top = np.argmax(rows_with_building)
bottom = height - np.argmax(rows_with_building[::-1])
building_height = bottom - top

# Diviser en 3 tiers
third_height = building_height // 3

# Upper
facade_upper = np.zeros_like(mask_array)
facade_upper[top:top+third_height, :] = mask_array[top:top+third_height, :]

# Lower
facade_lower = np.zeros_like(mask_array)
facade_lower[bottom-third_height:bottom, :] = mask_array[bottom-third_height:bottom, :]

# Combiner upper + lower
facade_mask = np.maximum(facade_upper, facade_lower)
facade_mask_img = Image.fromarray(facade_mask, mode="L")

upper_coverage = np.sum(facade_upper > 0) / (width * height)
lower_coverage = np.sum(facade_lower > 0) / (width * height)
combined_coverage = np.sum(facade_mask > 0) / (width * height)

print(f"✅ facade_upper: {upper_coverage*100:.1f}%")
print(f"✅ facade_lower: {lower_coverage*100:.1f}%")
print(f"✅ facade_mask (upper + lower): {combined_coverage*100:.1f}%")

# PASSE 3: Raffinement morphologique
print("\n" + "=" * 60)
print("PASSE 3: RAFFINEMENT MORPHOLOGIQUE")
print("=" * 60)

os.makedirs("output/morphological_refinement", exist_ok=True)

# Sauvegarder le masque original
facade_mask_img.save("output/morphological_refinement/01_original.png")

# Test avec différents paramètres
configs = [
    {
        "name": "clean_light",
        "operations": ["clean"],
        "strength": "light",
        "description": "Nettoyage léger (petits trous)"
    },
    {
        "name": "clean_medium",
        "operations": ["clean"],
        "strength": "medium",
        "description": "Nettoyage moyen"
    },
    {
        "name": "clean_feather",
        "operations": ["clean", "feather"],
        "strength": "medium",
        "description": "Nettoyage + bords adoucis"
    },
    {
        "name": "erode_dilate",
        "operations": ["erode", "dilate"],
        "strength": "light",
        "description": "Érosion puis dilatation (smooth edges)"
    },
    {
        "name": "clean_erode_dilate_feather",
        "operations": ["clean", "erode", "dilate", "feather"],
        "strength": "light",
        "description": "Pipeline complet"
    }
]

results = {}

for config in configs:
    print(f"\n🔄 Test: {config['name']}")
    print(f"   📋 {config['description']}")
    print(f"   ⚙️  Opérations: {' → '.join(config['operations'])}")
    
    # Raffiner avec les paramètres
    refined = refine_mask(
        mask=facade_mask_img,
        operations=config["operations"],
        strength=config["strength"]
    )
    
    # Sauvegarder
    filename = f"output/morphological_refinement/02_{config['name']}.png"
    refined.save(filename)
    
    # Statistiques
    refined_coverage = np.sum(np.array(refined) > 127) / (refined.size[0] * refined.size[1])
    diff = (refined_coverage - combined_coverage) * 100
    sign = "+" if diff > 0 else ""
    
    results[config['name']] = {
        "mask": refined,
        "coverage": refined_coverage,
        "diff": diff
    }
    
    print(f"   ✅ Couverture: {refined_coverage*100:.1f}% ({sign}{diff:.2f}%)")

# Comparaison
print("\n" + "=" * 60)
print("📊 COMPARAISON DES RÉSULTATS")
print("=" * 60)

print(f"\n{'Méthode':<35} {'Couverture':>12} {'Différence':>12}")
print("-" * 60)
print(f"{'Original (upper + lower)':<35} {combined_coverage*100:>11.2f}% {'':>12}")

for name, data in results.items():
    sign = "+" if data['diff'] > 0 else ""
    print(f"{name:<35} {data['coverage']*100:>11.2f}% {sign:>1}{data['diff']:>10.2f}%")

# Créer une visualisation comparative
print("\n🎨 Création de la visualisation...")

def create_overlay(base_img, mask, color, alpha=0.4):
    """Crée un overlay coloré"""
    overlay = base_img.copy().convert("RGBA")
    mask_array = np.array(mask)
    colored = Image.new("RGBA", base_img.size, color + (int(255 * alpha),))
    mask_rgba = Image.fromarray(mask_array).convert("L")
    result = Image.composite(colored, overlay, mask_rgba)
    return result

# Créer grille 3x2
w, h = image.size
comparison = Image.new("RGB", (w * 3, h * 2), (0, 0, 0))

# Ligne 1
comparison.paste(image, (0, 0))
vis_original = create_overlay(image, facade_mask_img, (255, 0, 0), 0.4).convert("RGB")
comparison.paste(vis_original, (w, 0))
vis_clean = create_overlay(image, results["clean_medium"]["mask"], (0, 255, 0), 0.4).convert("RGB")
comparison.paste(vis_clean, (w * 2, 0))

# Ligne 2
vis_feather = create_overlay(image, results["clean_feather"]["mask"], (0, 0, 255), 0.4).convert("RGB")
comparison.paste(vis_feather, (0, h))
vis_erode = create_overlay(image, results["erode_dilate"]["mask"], (255, 255, 0), 0.4).convert("RGB")
comparison.paste(vis_erode, (w, h))
vis_full = create_overlay(image, results["clean_erode_dilate_feather"]["mask"], (255, 0, 255), 0.4).convert("RGB")
comparison.paste(vis_full, (w * 2, h))

# Labels
draw = ImageDraw.Draw(comparison)
try:
    from PIL import ImageFont
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

labels = [
    (10, 10, "1. Original"),
    (w + 10, 10, "2. Original + upper/lower (rouge)"),
    (w * 2 + 10, 10, "3. Clean medium (vert)"),
    (10, h + 10, "4. Clean + feather (bleu)"),
    (w + 10, h + 10, "5. Erode + dilate (jaune)"),
    (w * 2 + 10, h + 10, "6. Pipeline complet (magenta)")
]

for x, y, text in labels:
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

comparison.save("output/morphological_refinement/comparison.png")
print("   ✅ Comparaison sauvegardée")

print("\n" + "=" * 60)
print("✅ CONCLUSION")
print("=" * 60)
print("""
🎯 Pour les façades architecturales:

✅ Raffinement morphologique:
   - Préserve la couverture globale
   - Nettoie les petits trous et artéfacts
   - Adoucit les bords
   - Traitement cohérent

❌ SAM2:
   - Réduit trop la zone (26% → 2%)
   - Cherche des objets isolés
   - Pas adapté aux scènes complexes

💡 RECOMMANDATION:
   Utiliser raffinement morphologique par défaut,
   SAM2 uniquement pour fenêtres/portes spécifiques

📁 Fichiers générés:
   - output/morphological_refinement/01_original.png
   - output/morphological_refinement/02_*.png (5 variantes)
   - output/morphological_refinement/comparison.png
""")
