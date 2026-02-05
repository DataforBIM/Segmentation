# Test du raffinement SAM2 sur le TARGET UNIQUEMENT
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import numpy as np
from segmentation.pipeline import segment_from_prompt
import os

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 70)
print("🎯 TEST: RAFFINEMENT SAM2 DU TARGET UNIQUEMENT")
print("=" * 70)
print("""
CONCEPT:
--------
SAM2 affine UNIQUEMENT l'objet sujet du prompt, pas toute la scène.

Exemple:
  Prompt: "change la façade en blanc moderne"
  
  PASSE 1 (OneFormer):
    building: 26% (toute la scène)
    
  PASSE 2 (SAM2):
    Raffine UNIQUEMENT le building (le target)
    Les fenêtres, portes (protected) restent sémantiques
    
  PASSE 3 (Fusion):
    facade_final = building_sam2_raffiné - (windows + doors)

👉 SAM2 n'intervient QUE sur le target (objet à modifier)
""")

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image: {image.size}")

os.makedirs("output/sam2_target_only", exist_ok=True)

# TEST 1: Sans raffinement SAM2 (baseline)
print("\n" + "=" * 70)
print("TEST 1: SANS RAFFINEMENT SAM2 (baseline)")
print("=" * 70)

result_without_sam2 = segment_from_prompt(
    image=image,
    user_prompt="change la façade en blanc moderne",
    refine_target_with_sam2=False,
    verbose=True
)

print(f"\n📊 Résultat sans SAM2:")
print(f"   Couverture: {result_without_sam2.coverage*100:.2f}%")

# Sauvegarder
result_without_sam2.final_mask.save("output/sam2_target_only/01_without_sam2.png")
result_without_sam2.target_mask.save("output/sam2_target_only/01_target_semantic.png")

# TEST 2: Avec raffinement SAM2 du target
print("\n" + "=" * 70)
print("TEST 2: AVEC RAFFINEMENT SAM2 DU TARGET")
print("=" * 70)

result_with_sam2 = segment_from_prompt(
    image=image,
    user_prompt="change la façade en blanc moderne",
    refine_target_with_sam2=True,
    verbose=True
)

print(f"\n📊 Résultat avec SAM2:")
print(f"   Couverture: {result_with_sam2.coverage*100:.2f}%")

# Sauvegarder
result_with_sam2.final_mask.save("output/sam2_target_only/02_with_sam2.png")
result_with_sam2.target_mask.save("output/sam2_target_only/02_target_sam2_refined.png")

# Comparaison
print("\n" + "=" * 70)
print("📊 COMPARAISON")
print("=" * 70)

coverage_diff = (result_with_sam2.coverage - result_without_sam2.coverage) * 100
sign = "+" if coverage_diff > 0 else ""

print(f"\n{'Méthode':<30} {'Couverture':>15} {'Différence':>15}")
print("-" * 70)
print(f"{'OneFormer seul':<30} {result_without_sam2.coverage*100:>14.2f}% {'':>15}")
print(f"{'OneFormer + SAM2 target':<30} {result_with_sam2.coverage*100:>14.2f}% {sign:>1}{coverage_diff:>13.2f}%")

# Créer visualisation comparative
print("\n🎨 Création de la visualisation...")

def create_overlay(base_img, mask, color, alpha=0.5):
    """Crée un overlay coloré"""
    overlay = base_img.copy().convert("RGBA")
    mask_array = np.array(mask)
    colored = Image.new("RGBA", base_img.size, color + (int(255 * alpha),))
    mask_rgba = Image.fromarray(mask_array).convert("L")
    result = Image.composite(colored, overlay, mask_rgba)
    return result

# Créer grille 2x3
w, h = image.size
comparison = Image.new("RGB", (w * 2, h * 3), (20, 20, 20))

# Ligne 1: Images originales
comparison.paste(image, (0, 0))
comparison.paste(image, (w, 0))

# Ligne 2: Targets
target_without = create_overlay(image, result_without_sam2.target_mask, (255, 0, 0), 0.5).convert("RGB")
comparison.paste(target_without, (0, h))
target_with = create_overlay(image, result_with_sam2.target_mask, (0, 255, 0), 0.5).convert("RGB")
comparison.paste(target_with, (w, h))

# Ligne 3: Final masks
final_without = create_overlay(image, result_without_sam2.final_mask, (255, 255, 0), 0.5).convert("RGB")
comparison.paste(final_without, (0, h * 2))
final_with = create_overlay(image, result_with_sam2.final_mask, (0, 255, 255), 0.5).convert("RGB")
comparison.paste(final_with, (w, h * 2))

# Labels
draw = ImageDraw.Draw(comparison)
try:
    from PIL import ImageFont
    font = ImageFont.truetype("arial.ttf", 28)
    font_small = ImageFont.truetype("arial.ttf", 22)
except:
    font = ImageFont.load_default()
    font_small = font

labels = [
    (10, 10, "SANS RAFFINEMENT SAM2", font),
    (w + 10, 10, "AVEC RAFFINEMENT SAM2", font),
    (10, h + 10, "Target (OneFormer seul)", font_small),
    (w + 10, h + 10, "Target (OneFormer + SAM2)", font_small),
    (10, h * 2 + 10, "Final (après protection)", font_small),
    (w + 10, h * 2 + 10, "Final (après protection)", font_small)
]

for x, y, text, f in labels:
    # Ombre
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0), font=f)
    # Texte
    draw.text((x, y), text, fill=(255, 255, 255), font=f)

comparison.save("output/sam2_target_only/comparison.png")
print("   ✅ Comparaison sauvegardée")

print("\n" + "=" * 70)
print("✅ CONCLUSION")
print("=" * 70)
print(f"""
🎯 RAFFINEMENT SAM2 DU TARGET UNIQUEMENT

PRINCIPE:
  • OneFormer détecte la scène globalement (building: 26%)
  • SAM2 affine UNIQUEMENT le building (le target du prompt)
  • Les autres éléments (protected, context) restent sémantiques

RÉSULTATS:
  • Sans SAM2: {result_without_sam2.coverage*100:.2f}% de couverture
  • Avec SAM2: {result_with_sam2.coverage*100:.2f}% de couverture
  • Différence: {sign}{coverage_diff:.2f}%

AVANTAGES:
  ✓ SAM2 se concentre sur l'objet à modifier
  ✓ Bords plus précis du target
  ✓ Reste cohérent avec la scène
  ✓ Pas de sur-segmentation

📁 Fichiers générés:
   - output/sam2_target_only/01_without_sam2.png
   - output/sam2_target_only/01_target_semantic.png
   - output/sam2_target_only/02_with_sam2.png
   - output/sam2_target_only/02_target_sam2_refined.png
   - output/sam2_target_only/comparison.png

💡 UTILISATION:
   result = segment_from_prompt(
       image=image,
       user_prompt="change la façade en blanc",
       refine_target_with_sam2=True  # Active le raffinement SAM2 du target
   )
""")
