# Test des masques de transition
import sys
import numpy as np
from PIL import Image, ImageDraw

# Créer une image de test
print("=" * 70)
print("🎨 TEST: MASQUES DE TRANSITION (BLENDING PROGRESSIF)")
print("=" * 70)
print()

# 1. Créer image de test
print("📝 Étape 1: Création d'une image et masque de test")
width, height = 512, 512
test_image = Image.new("RGB", (width, height), (200, 220, 240))

# Dessiner un jardin simple
draw = ImageDraw.Draw(test_image)
# Ciel
draw.rectangle([(0, 0), (width, height//2)], fill=(135, 206, 235))
# Herbe
draw.rectangle([(0, height//2), (width, height)], fill=(124, 252, 0))
# Quelques arbres
for x in [100, 300, 450]:
    draw.ellipse([(x-30, height//2-50), (x+30, height//2+10)], fill=(34, 139, 34))

test_image.save("output/test_original.png")
print("   ✅ Image test créée: output/test_original.png")

# 2. Créer un masque central
print("\n📝 Étape 2: Création du masque core")
mask_core = Image.new("L", (width, height), 0)
draw_mask = ImageDraw.Draw(mask_core)
# Zone circulaire au centre (pour ajouter des fleurs)
center_x, center_y = width // 2, int(height * 0.7)
radius = 80
draw_mask.ellipse([
    (center_x - radius, center_y - radius),
    (center_x + radius, center_y + radius)
], fill=255)

mask_core.save("output/test_mask_core.png")
print("   ✅ Masque core créé: output/test_mask_core.png")

# 3. Créer les masques de transition
print("\n📝 Étape 3: Génération des masques de transition")
from segmentation.transition_masks import (
    create_transition_masks,
    visualize_transition_masks,
    create_mask_comparison,
    compute_adaptive_transition_width
)

# Test différentes largeurs
transition_configs = [
    {"width": 6, "type": "linear", "name": "Narrow Linear"},
    {"width": 12, "type": "cosine", "name": "Medium Cosine"},
    {"width": 20, "type": "gaussian", "name": "Wide Gaussian"},
    {"width": "auto", "type": "cosine", "name": "Adaptive Cosine"}
]

for config in transition_configs:
    print(f"\n   🔧 Test: {config['name']}")
    
    if config["width"] == "auto":
        width_value = compute_adaptive_transition_width(mask_core, test_image.size)
        print(f"      → Largeur adaptative calculée: {width_value}px")
    else:
        width_value = config["width"]
    
    # Créer masques
    trans_masks = create_transition_masks(
        mask_core=mask_core,
        transition_width=width_value,
        gradient_type=config["type"],
        feather_strength=0.5
    )
    
    # Visualiser
    prefix = config["name"].lower().replace(" ", "_")
    
    # Preview avec overlay
    preview = visualize_transition_masks(
        test_image,
        trans_masks,
        save_path=f"output/transition_{prefix}_preview.png"
    )
    print(f"      ✅ Preview: output/transition_{prefix}_preview.png")
    
    # Comparaison côte à côte
    comparison = create_mask_comparison(
        trans_masks,
        save_path=f"output/transition_{prefix}_masks.png"
    )
    print(f"      ✅ Masks: output/transition_{prefix}_masks.png")
    
    # Stats
    core_pixels = np.sum(np.array(trans_masks.core) > 127)
    transition_pixels = np.sum(np.array(trans_masks.transition) > 50)
    combined_pixels = np.sum(np.array(trans_masks.combined) > 127)
    
    print(f"      📊 Core: {core_pixels} px")
    print(f"      📊 Transition: {transition_pixels} px")
    print(f"      📊 Combined: {combined_pixels} px")
    print(f"      📊 Expansion: {(combined_pixels/core_pixels - 1)*100:.1f}%")

# 4. Test du blending
print("\n📝 Étape 4: Test du blending avec transition")
from segmentation.transition_masks import blend_with_transition

# Créer une "image générée" (fleurs rouges)
generated_image = test_image.copy()
draw_gen = ImageDraw.Draw(generated_image)
# Zone avec fleurs rouges
for offset in [(-20, -15), (0, 0), (20, 15), (-15, 20), (18, -18)]:
    x, y = center_x + offset[0], center_y + offset[1]
    draw_gen.ellipse([
        (x - 8, y - 8),
        (x + 8, y + 8)
    ], fill=(255, 50, 80))

generated_image.save("output/test_generated.png")
print("   ✅ Image générée créée: output/test_generated.png")

# Blender avec transition
trans_masks_blend = create_transition_masks(
    mask_core=mask_core,
    transition_width=12,
    gradient_type="cosine",
    feather_strength=0.5
)

blended = blend_with_transition(
    original_image=test_image,
    generated_image=generated_image,
    transition_masks=trans_masks_blend
)

blended.save("output/test_blended_result.png")
print("   ✅ Résultat blendé: output/test_blended_result.png")

# Comparaison: sans transition vs avec transition
print("\n📝 Étape 5: Comparaison avec/sans transition")

# Sans transition (cut brutal)
no_transition = test_image.copy()
mask_array = np.array(mask_core) > 127
gen_array = np.array(generated_image)
orig_array = np.array(no_transition)

for y in range(height):
    for x in range(width):
        if mask_array[y, x]:
            orig_array[y, x] = gen_array[y, x]

no_transition_result = Image.fromarray(orig_array)
no_transition_result.save("output/test_no_transition.png")
print("   ✅ Sans transition: output/test_no_transition.png")

print("\n" + "=" * 70)
print("✅ TESTS TERMINÉS")
print("=" * 70)
print("\n📊 RÉSULTATS:")
print("   • output/test_original.png - Image originale")
print("   • output/test_generated.png - Image avec fleurs")
print("   • output/test_no_transition.png - Blend brutal (AVANT)")
print("   • output/test_blended_result.png - Blend progressif (APRÈS) ✨")
print("\n🎨 MASQUES DE TRANSITION:")
print("   • output/transition_*_preview.png - Overlays colorés")
print("   • output/transition_*_masks.png - Comparaisons masques")
print("\n💡 INTERPRÉTATION:")
print("   🔴 Rouge = Core (100% généré)")
print("   🟡 Jaune = Transition (gradient 100%→0%)")
print("   ⚪ Transparent = Original (0% généré)")
print()
