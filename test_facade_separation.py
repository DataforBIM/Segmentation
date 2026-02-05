# Test de séparation façade / ouvertures
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import semantic_segment, prepare_facade_masks
import os

# URL de l'image de test (façade)
IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 60)
print("🏛️  TEST SÉPARATION FAÇADE / OUVERTURES")
print("=" * 60)

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image chargée: {image.size}")

# Segmentation sémantique avec OneFormer
print("\n🔷 Segmentation avec OneFormer...")
semantic_map = semantic_segment(image, model_type="oneformer")

# Préparer les masques de façade avec séparation des ouvertures
print("\n🔧 Séparation façade / ouvertures...")
facade_masks = prepare_facade_masks(semantic_map, image.size)

# Afficher les résultats
print(f"\n📦 Masques générés:")
for name, mask in facade_masks.items():
    if mask is not None:
        coverage = np.sum(np.array(mask) > 0) / (mask.size[0] * mask.size[1])
        print(f"   - {name}: {coverage*100:.1f}%")

# Sauvegarder tous les masques
print("\n💾 Sauvegarde des masques...")
os.makedirs("output/facade_separation", exist_ok=True)

for name, mask in facade_masks.items():
    if mask is not None:
        output_path = f"output/facade_separation/{name}.png"
        mask.save(output_path)
        print(f"   ✓ {output_path}")

# Créer une visualisation comparative
print("\n🎨 Création de la visualisation comparative...")

def create_overlay(base_image, mask, color, alpha=0.5):
    """Crée un overlay coloré sur l'image"""
    overlay = base_image.copy().convert("RGBA")
    mask_array = np.array(mask)
    
    colored = Image.new("RGBA", base_image.size, color + (int(255 * alpha),))
    mask_rgba = Image.fromarray(mask_array).convert("L")
    
    result = Image.composite(colored, overlay, mask_rgba)
    return result

# Visualisation 1: Façade complète (avec fenêtres)
if facade_masks["facade_full"]:
    vis1 = create_overlay(image, facade_masks["facade_full"], (255, 0, 0), 0.3)
    vis1.convert("RGB").save("output/facade_separation/vis_01_facade_full.png")
    print("   ✓ Visualisation 1: Façade complète (rouge)")

# Visualisation 2: Fenêtres/Portes protégées
if facade_masks["protected"]:
    vis2 = create_overlay(image, facade_masks["protected"], (0, 255, 0), 0.5)
    vis2.convert("RGB").save("output/facade_separation/vis_02_protected.png")
    print("   ✓ Visualisation 2: Ouvertures protégées (vert)")

# Visualisation 3: Façade nettoyée (SANS fenêtres)
if facade_masks["facade_clean"]:
    vis3 = create_overlay(image, facade_masks["facade_clean"], (0, 0, 255), 0.3)
    vis3.convert("RGB").save("output/facade_separation/vis_03_facade_clean.png")
    print("   ✓ Visualisation 3: Façade nettoyée (bleu)")

# Visualisation 4: Comparaison côte à côte
if facade_masks["facade_full"] and facade_masks["facade_clean"] and facade_masks["protected"]:
    # Créer une image 2x2
    w, h = image.size
    comparison = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
    
    # Original
    comparison.paste(image, (0, 0))
    
    # Façade complète
    vis_full = create_overlay(image, facade_masks["facade_full"], (255, 0, 0), 0.4).convert("RGB")
    comparison.paste(vis_full, (w, 0))
    
    # Ouvertures protégées
    vis_prot = create_overlay(image, facade_masks["protected"], (0, 255, 0), 0.6).convert("RGB")
    comparison.paste(vis_prot, (0, h))
    
    # Façade nettoyée
    vis_clean = create_overlay(image, facade_masks["facade_clean"], (0, 0, 255), 0.4).convert("RGB")
    comparison.paste(vis_clean, (w, h))
    
    # Ajouter des labels
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "1. Original", fill=(255, 255, 255), font=font)
    draw.text((w + 10, 10), "2. Façade complète (rouge)", fill=(255, 255, 255), font=font)
    draw.text((10, h + 10), "3. Ouvertures protégées (vert)", fill=(255, 255, 255), font=font)
    draw.text((w + 10, h + 10), "4. Façade SANS ouvertures (bleu)", fill=(255, 255, 255), font=font)
    
    comparison.save("output/facade_separation/comparison.png")
    print("   ✓ Visualisation 4: Comparaison complète")

print("\n" + "=" * 60)
print("✅ RÉSULTAT:")
print("=" * 60)
print("""
📍 Concept démontré:

1️⃣  facade_full = façade complète (avec fenêtres incluses)
2️⃣  protected = fenêtres + portes (à NE PAS modifier)
3️⃣  facade_clean = facade_full - protected

👉 UTILISATION POUR INPAINTING:

# Changer la couleur de la façade SANS toucher aux vitres
result = inpaint(
    image=image,
    mask=facade_masks["facade_clean"],  # ← Façade SANS fenêtres
    prompt="white modern facade"
)

✅ Avantages:
   - Aucun reflet de vitre cassé
   - Aucun cadre repeint
   - Séparation nette façade/ouvertures

📁 Fichiers générés:
   - output/facade_separation/facade_clean.png
   - output/facade_separation/protected.png
   - output/facade_separation/comparison.png
""")
