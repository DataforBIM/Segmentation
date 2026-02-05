# Test interactif de segmentation
from PIL import Image
import requests
from io import BytesIO
from segmentation.pipeline import segment_from_prompt
import numpy as np
import os

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("🎯 TEST DE SEGMENTATION INTERACTIF\n")

# Charger l'image
print("📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"✅ Image chargée: {image.size}\n")

os.makedirs("output/quick_test", exist_ok=True)

# Prompt par défaut
default_prompt = "change la façade en blanc moderne"

print("💬 Entrez votre prompt (ou appuyez sur Entrée pour utiliser le défaut):")
print(f"   Défaut: \"{default_prompt}\"")
user_input = input("➜ ").strip()

prompt = user_input if user_input else default_prompt

print(f"\n{'=' * 70}")
print(f"🔄 Segmentation en cours...")
print(f"{'=' * 70}\n")

# Segmentation
result = segment_from_prompt(
    image=image,
    user_prompt=prompt,
    refine_target_with_sam2=False,
    verbose=True
)

# Statistiques
w, h = image.size
total_pixels = w * h

target_pct = np.sum(np.array(result.target_mask) > 127) / total_pixels * 100
protected_pct = np.sum(np.array(result.protected_mask) > 127) / total_pixels * 100
final_pct = np.sum(np.array(result.final_mask) > 127) / total_pixels * 100

print(f"\n{'=' * 70}")
print("📊 RÉSULTATS")
print("=" * 70)
print(f"""
Target:     {target_pct:6.2f}%  (zone à modifier)
Protected:  {protected_pct:6.2f}%  (zones protégées)
Final:      {final_pct:6.2f}%  (zone finale = target - protected)

Équation: {target_pct:.2f}% - {protected_pct:.2f}% = {final_pct:.2f}%

Classes détectées:
  • Target:    {', '.join(result.target.primary)}
  • Protected: {', '.join(result.target.protected)}
""")

# Sauvegarder
result.target_mask.save("output/quick_test/target.png")
result.protected_mask.save("output/quick_test/protected.png")
result.final_mask.save("output/quick_test/final.png")

# Visualisation
from PIL import ImageDraw, ImageFont

def create_overlay(base, mask, color, alpha=0.6):
    overlay = base.copy().convert("RGBA")
    mask_array = np.array(mask)
    colored = Image.new("RGBA", base.size, color + (int(255 * alpha),))
    mask_rgba = Image.fromarray(mask_array).convert("L")
    return Image.composite(colored, overlay, mask_rgba).convert("RGB")

# Créer comparaison
comparison = Image.new("RGB", (w * 2, h * 2), (30, 30, 30))
comparison.paste(image, (0, 0))
comparison.paste(create_overlay(image, result.target_mask, (255, 165, 0), 0.5), (w, 0))
comparison.paste(create_overlay(image, result.protected_mask, (255, 0, 0), 0.6), (0, h))
comparison.paste(create_overlay(image, result.final_mask, (0, 255, 0), 0.5), (w, h))

# Labels
draw = ImageDraw.Draw(comparison)
try:
    font = ImageFont.truetype("arial.ttf", 32)
except:
    font = ImageFont.load_default()

def draw_label(xy, text, bg):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0]-8, bbox[1]-8, bbox[2]+8, bbox[3]+8], fill=bg+(200,))
    draw.text((x, y), text, fill=(255,255,255), font=font)

draw_label((20, 20), "ORIGINAL", (50, 50, 50))
draw_label((w+20, 20), f"TARGET ({target_pct:.1f}%)", (255, 140, 0))
draw_label((20, h+20), f"PROTECTED ({protected_pct:.1f}%)", (200, 0, 0))
draw_label((w+20, h+20), f"FINAL ({final_pct:.1f}%)", (0, 150, 0))

comparison.save("output/quick_test/comparison.png")

print("💾 Fichiers sauvegardés:")
print("   • output/quick_test/target.png")
print("   • output/quick_test/protected.png")
print("   • output/quick_test/final.png")
print("   • output/quick_test/comparison.png")
print("\n✅ Test terminé!")
