"""
Test du feathering adaptatif sur les masques de transition
Vérifie que le mask_transition reçoit un feathering dynamique
"""

import numpy as np
from PIL import Image, ImageDraw
import os
from segmentation.transition_masks import create_transition_masks

# Configuration
OUTPUT_DIR = "output/transition_feather_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_test_mask(size: tuple, mask_type: str) -> Image.Image:
    """Crée un masque de test"""
    w, h = size
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "small":
        # Petit cercle (5%)
        radius = int(min(w, h) * 0.1)
        draw.ellipse(
            [(w//2 - radius, h//2 - radius), (w//2 + radius, h//2 + radius)],
            fill=255
        )
    elif mask_type == "medium":
        # Cercle moyen (20%)
        radius = int(min(w, h) * 0.25)
        draw.ellipse(
            [(w//2 - radius, h//2 - radius), (w//2 + radius, h//2 + radius)],
            fill=255
        )
    elif mask_type == "large":
        # Grand rectangle (50%)
        margin_w = int(w * 0.25)
        margin_h = int(h * 0.25)
        draw.rectangle(
            [(margin_w, margin_h), (w - margin_w, h - margin_h)],
            fill=255
        )
    
    return mask


def visualize_transition_with_feather(
    mask_core: Image.Image,
    masks: dict,
    title: str,
    feather_info: str
):
    """Visualise les 3 masques côte à côte"""
    w, h = mask_core.size
    
    # Canvas 3x plus large
    canvas = Image.new('RGB', (w * 3, h + 100), (40, 40, 40))
    
    # Convertir masques en RGB
    for i, (name, mask) in enumerate([
        ("Core", masks["core"]),
        ("Transition", masks["transition"]),
        ("Combined", masks["combined"])
    ]):
        color = (255, 100, 100) if name == "Core" else (100, 255, 100) if name == "Transition" else (100, 100, 255)
        mask_rgb = Image.new('RGB', (w, h), (0, 0, 0))
        mask_rgb.paste(Image.new('RGB', (w, h), color), mask=mask)
        canvas.paste(mask_rgb, (i * w, 100))
        
        # Label
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        draw.text((i * w + 10, 75), name, fill=(200, 200, 200))
    
    # Titre et info
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    draw.text((10, 35), feather_info, fill=(150, 255, 150))
    
    return canvas


def test_adaptive_feathering_on_transitions():
    """Test du feathering adaptatif sur différents masques de transition"""
    
    print("🔬 TEST FEATHERING ADAPTATIF SUR TRANSITIONS\n")
    print("=" * 70)
    
    # Configurations
    test_configs = [
        ((512, 512), "small", "512px_small_transition"),
        ((512, 512), "medium", "512px_medium_transition"),
        ((512, 512), "large", "512px_large_transition"),
        ((1024, 1024), "small", "1024px_small_transition"),
        ((1024, 1024), "medium", "1024px_medium_transition"),
        ((1024, 1024), "large", "1024px_large_transition"),
        ((2048, 2048), "small", "2048px_small_transition"),
        ((2048, 2048), "medium", "2048px_medium_transition"),
    ]
    
    results = []
    
    for size, mask_type, name in test_configs:
        print(f"\n📊 {name}")
        print(f"   Résolution: {size[0]}x{size[1]}")
        print(f"   Type: {mask_type}")
        
        # Créer masque core
        mask_core = create_test_mask(size, mask_type)
        
        # Stats du core
        core_array = np.array(mask_core)
        core_pixels = np.sum(core_array > 127)
        core_coverage = core_pixels / (size[0] * size[1])
        
        print(f"   Core coverage: {core_coverage*100:.1f}%")
        print(f"   Core pixels: {core_pixels:,}")
        
        # ✨ AVEC feathering adaptatif
        trans_adaptive = create_transition_masks(
            mask_core=mask_core,
            transition_width=16,
            gradient_type="cosine",
            adaptive_feather=True
        )
        
        # ❌ SANS feathering adaptatif (fixe)
        trans_fixed = create_transition_masks(
            mask_core=mask_core,
            transition_width=16,
            gradient_type="cosine",
            adaptive_feather=False,
            feather_strength=0.5
        )
        
        # Analyser le masque de transition
        trans_adaptive_array = np.array(trans_adaptive.transition)
        trans_fixed_array = np.array(trans_fixed.transition)
        
        # Calculer la "douceur" (écart-type des valeurs dans la transition)
        trans_adaptive_std = np.std(trans_adaptive_array[trans_adaptive_array > 0])
        trans_fixed_std = np.std(trans_fixed_array[trans_fixed_array > 0])
        
        # Compter pixels de transition
        trans_adaptive_pixels = np.sum(trans_adaptive_array > 0)
        trans_fixed_pixels = np.sum(trans_fixed_array > 0)
        
        print(f"   ✨ Adaptive transition std: {trans_adaptive_std:.2f}")
        print(f"   ❌ Fixed transition std: {trans_fixed_std:.2f}")
        print(f"   Différence douceur: {(trans_adaptive_std - trans_fixed_std):.2f}")
        
        results.append({
            'name': name,
            'size': size,
            'type': mask_type,
            'core_coverage': core_coverage,
            'adaptive_std': trans_adaptive_std,
            'fixed_std': trans_fixed_std,
            'diff': trans_adaptive_std - trans_fixed_std
        })
        
        # Visualiser
        vis_adaptive = visualize_transition_with_feather(
            mask_core,
            {
                "core": trans_adaptive.core,
                "transition": trans_adaptive.transition,
                "combined": trans_adaptive.combined
            },
            f"{name} - ADAPTIVE",
            f"Transition STD: {trans_adaptive_std:.2f} | Feathering: ADAPTATIF"
        )
        
        vis_fixed = visualize_transition_with_feather(
            mask_core,
            {
                "core": trans_fixed.core,
                "transition": trans_fixed.transition,
                "combined": trans_fixed.combined
            },
            f"{name} - FIXED",
            f"Transition STD: {trans_fixed_std:.2f} | Feathering: FIXE (0.5)"
        )
        
        # Sauvegarder (réduire si trop grand)
        if size[0] > 1024:
            vis_adaptive = vis_adaptive.resize((1536, int(1536 * vis_adaptive.size[1] / vis_adaptive.size[0])))
            vis_fixed = vis_fixed.resize((1536, int(1536 * vis_fixed.size[1] / vis_fixed.size[0])))
        
        vis_adaptive.save(f"{OUTPUT_DIR}/{name}_adaptive.png")
        vis_fixed.save(f"{OUTPUT_DIR}/{name}_fixed.png")
    
    print("\n" + "=" * 70)
    print("📈 RÉSUMÉ\n")
    
    print(f"{'Configuration':<30} {'Résolution':<12} {'Coverage':<10} {'Adaptive STD':<13} {'Fixed STD':<10} {'Diff':<8}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['name']:<30} {r['size'][0]}x{r['size'][1]:<6} {r['core_coverage']*100:>6.1f}%   {r['adaptive_std']:>9.2f}    {r['fixed_std']:>7.2f}   {r['diff']:>6.2f}")
    
    print("\n✅ Visualisations sauvegardées dans:", OUTPUT_DIR)
    
    # Vérifications
    print("\n🔍 VÉRIFICATIONS:")
    
    # Le feathering adaptatif devrait être plus doux (STD plus élevé) pour grandes zones
    large_masks = [r for r in results if r['type'] == 'large']
    if large_masks:
        avg_diff_large = sum(r['diff'] for r in large_masks) / len(large_masks)
        if avg_diff_large > 0:
            print(f"   ✅ Grandes zones : adaptatif plus doux (avg diff: {avg_diff_large:.2f})")
        else:
            print(f"   ⚠️ Grandes zones : pas de différence significative")
    
    # Le feathering adaptatif devrait s'adapter à la résolution
    adaptive_512 = [r for r in results if r['size'][0] == 512 and r['type'] == 'medium']
    adaptive_1024 = [r for r in results if r['size'][0] == 1024 and r['type'] == 'medium']
    adaptive_2048 = [r for r in results if r['size'][0] == 2048 and r['type'] == 'medium']
    
    if adaptive_512 and adaptive_1024:
        if adaptive_1024[0]['adaptive_std'] > adaptive_512[0]['adaptive_std']:
            print(f"   ✅ Résolution adaptive: 1024px plus doux que 512px")
        else:
            print(f"   ⚠️ Résolution: pas d'augmentation avec résolution")


if __name__ == "__main__":
    test_adaptive_feathering_on_transitions()
