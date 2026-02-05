# Test du raffinement SAM2 sur les masques sémantiques
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import (
    semantic_segment,
    sample_points_from_mask,
    refine_mask_with_sam2,
    subtract_masks
)
from models.sam2 import load_sam2
import os

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 60)
print("🎯 TEST RAFFINEMENT SAM2")
print("=" * 60)
print("""
PIPELINE EN 3 PASSES:

PASSE 1: OneFormer (sémantique)
         → Détection globale "building"

PASSE 2: Division verticale
         → facade_upper, facade_middle, facade_lower

PASSE 3: SAM2 (raffinement)
         → Bords précis, découpes fines
         → Nettoyage zones ambiguës
""")

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image: {image.size}")

# PASSE 1: Segmentation sémantique OneFormer
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
facade_upper_img = Image.fromarray(facade_upper, mode="L")

# Lower
facade_lower = np.zeros_like(mask_array)
facade_lower[bottom-third_height:bottom, :] = mask_array[bottom-third_height:bottom, :]
facade_lower_img = Image.fromarray(facade_lower, mode="L")

# Combiner upper + lower
facade_mask = np.maximum(facade_upper, facade_lower)
facade_mask_img = Image.fromarray(facade_mask, mode="L")

upper_coverage = np.sum(facade_upper > 0) / (width * height)
lower_coverage = np.sum(facade_lower > 0) / (width * height)
combined_coverage = np.sum(facade_mask > 0) / (width * height)

print(f"✅ facade_upper: {upper_coverage*100:.1f}%")
print(f"✅ facade_lower: {lower_coverage*100:.1f}%")
print(f"✅ facade_mask (upper + lower): {combined_coverage*100:.1f}%")

# PASSE 3: Raffinement SAM2
print("\n" + "=" * 60)
print("PASSE 3: RAFFINEMENT SAM2")
print("=" * 60)
print("""
Approche: Raffiner le masque 'building' COMPLET avec SAM2,
          puis découper upper + lower depuis le résultat raffiné.
          
          Cela évite de segmenter des régions fragmentées.
""")

# Charger SAM2
print("🧠 Chargement de SAM2...")
sam2_model, sam2_processor = load_sam2()
print("   ✅ SAM2 chargé")

# Test avec différentes stratégies
strategies = ["grid", "random", "center"]
results = {}

os.makedirs("output/sam2_refinement", exist_ok=True)

for strategy in strategies:
    print(f"\n🔄 Test stratégie: {strategy}")
    
    # Raffiner le masque building COMPLET (pas la façade fragmentée)
    print(f"   Raffinage du masque building complet...")
    refined_building = refine_mask_with_sam2(
        image=image,
        semantic_mask=building_mask,
        sam2_model=sam2_model,
        sam2_processor=sam2_processor,
        num_points=20,  # Plus de points pour un grand objet
        strategy=strategy
    )
    
    # Découper upper + lower depuis le building raffiné
    refined_array = np.array(refined_building)
    
    # Appliquer la même division verticale
    refined_upper = np.zeros_like(refined_array)
    refined_upper[top:top+third_height, :] = refined_array[top:top+third_height, :]
    
    refined_lower = np.zeros_like(refined_array)
    refined_lower[bottom-third_height:bottom, :] = refined_array[bottom-third_height:bottom, :]
    
    # Combiner
    refined_facade = np.maximum(refined_upper, refined_lower)
    refined_mask = Image.fromarray(refined_facade, mode="L")
    
    # Visualiser les points (échantillonnés depuis building_mask)
    points = sample_points_from_mask(building_mask, num_points=20, strategy=strategy)
    print(f"   📍 {len(points)} points échantillonnés")
    
    # Visualiser les points
    # Visualiser les points (échantillonnés depuis building_mask)
    points = sample_points_from_mask(building_mask, num_points=20, strategy=strategy)
    print(f"   📍 {len(points)} points échantillonnés")
    
    vis_points = image.copy()
    draw = ImageDraw.Draw(vis_points)
    for x, y in points:
        draw.ellipse([x-5, y-5, x+5, y+5], fill=(255, 0, 0), outline=(255, 255, 255))
    vis_points.save(f"output/sam2_refinement/points_{strategy}.png")
    
    # Sauvegarder le masque raffiné
    refined_mask.save(f"output/sam2_refinement/refined_{strategy}.png")
    
    # Statistiques
    refined_coverage = np.sum(np.array(refined_mask) > 127) / (image.size[0] * image.size[1])
    results[strategy] = {
        "mask": refined_mask,
        "coverage": refined_coverage,
        "points": points
    }
    
    print(f"   ✅ Couverture raffinée: {refined_coverage*100:.1f}%")

# Comparaison
print("\n" + "=" * 60)
print("📊 COMPARAISON DES RÉSULTATS")
print("=" * 60)

print(f"\n{'Méthode':<20} {'Couverture':>12} {'Différence':>12}")
print("-" * 45)
print(f"{'OneFormer (base)':<20} {combined_coverage*100:>11.2f}% {'':>12}")

for strategy, data in results.items():
    diff = (data['coverage'] - combined_coverage) * 100
    sign = "+" if diff > 0 else ""
    print(f"{'SAM2 ' + strategy:<20} {data['coverage']*100:>11.2f}% {sign:>1}{diff:>10.2f}%")

# Créer une visualisation comparative
print("\n🎨 Création de la visualisation comparative...")

def create_overlay(base_img, mask, color, alpha=0.4):
    """Crée un overlay coloré"""
    overlay = base_img.copy().convert("RGBA")
    mask_array = np.array(mask)
    colored = Image.new("RGBA", base_img.size, color + (int(255 * alpha),))
    mask_rgba = Image.fromarray(mask_array).convert("L")
    result = Image.composite(colored, overlay, mask_rgba)
    return result

# Créer grille 2x2
w, h = image.size
comparison = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))

# Original
comparison.paste(image, (0, 0))

# OneFormer seul
vis_oneformer = create_overlay(image, facade_mask_img, (255, 0, 0), 0.4).convert("RGB")
comparison.paste(vis_oneformer, (w, 0))

# SAM2 grid
vis_sam2_grid = create_overlay(image, results["grid"]["mask"], (0, 255, 0), 0.4).convert("RGB")
comparison.paste(vis_sam2_grid, (0, h))

# SAM2 center
vis_sam2_center = create_overlay(image, results["center"]["mask"], (0, 0, 255), 0.4).convert("RGB")
comparison.paste(vis_sam2_center, (w, h))

# Labels
draw = ImageDraw.Draw(comparison)
try:
    from PIL import ImageFont
    font = ImageFont.truetype("arial.ttf", 30)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "1. Original", fill=(255, 255, 255), font=font)
draw.text((w + 10, 10), "2. OneFormer seul (rouge)", fill=(255, 255, 255), font=font)
draw.text((10, h + 10), "3. OneFormer + SAM2 grid (vert)", fill=(255, 255, 255), font=font)
draw.text((w + 10, h + 10), "4. OneFormer + SAM2 center (bleu)", fill=(255, 255, 255), font=font)

comparison.save("output/sam2_refinement/comparison.png")
print("   ✅ Comparaison sauvegardée")

print("\n" + "=" * 60)
print("✅ RÉSULTAT")
print("=" * 60)
print("""
🎯 SAM2 affine les bords détectés par OneFormer:

1. Bords de façade plus précis
2. Découpes fines autour des éléments
3. Nettoyage des zones ambiguës (ombres, végétation)

👉 SAM2 n'invente rien, il AFFINE ce que OneFormer a trouvé

📁 Fichiers générés:
   - output/sam2_refinement/points_*.png (visualisation des points)
   - output/sam2_refinement/refined_*.png (masques raffinés)
   - output/sam2_refinement/comparison.png (comparaison 2x2)
""")

print("\n💡 UTILISATION:")
print("""
# Dans votre pipeline:
semantic_map = semantic_segment(image, model_type="oneformer")
facade_mask = semantic_map.masks["building"]

# Raffiner avec SAM2
facade_refined = refine_mask_with_sam2(
    image=image,
    semantic_mask=facade_mask,
    num_points=10,
    strategy="grid"
)

# Utiliser le masque raffiné pour l'inpainting
result = inpaint(image, facade_refined, "white modern facade")
""")
