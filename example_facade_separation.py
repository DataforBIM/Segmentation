# =====================================================
# EXEMPLE: UTILISATION DES MASQUES FAÇADE/OUVERTURES
# =====================================================
# Démontre comment utiliser prepare_facade_masks() dans un pipeline

"""
CONCEPT:
--------
Problème classique: les fenêtres sont partiellement incluses dans la façade
lors de la segmentation.

Solution: Séparation explicite façade / ouvertures

ÉTAPES:
-------
1. Segmentation sémantique (OneFormer)
2. Extraction des masques architecturaux
3. Séparation façade / ouvertures
4. Utilisation pour inpainting ciblé
"""

from PIL import Image
from segmentation import (
    semantic_segment,
    prepare_facade_masks,
    load_oneformer
)

# =====================================================
# EXEMPLE 1: Changer la couleur de la façade
# =====================================================

def example_change_facade_color(image_path: str):
    """
    Change la couleur de la façade SANS toucher aux fenêtres/portes
    """
    
    # 1. Charger l'image
    image = Image.open(image_path)
    
    # 2. Segmentation sémantique
    semantic_map = semantic_segment(image, model_type="oneformer")
    
    # 3. Préparer les masques avec séparation
    facade_masks = prepare_facade_masks(semantic_map, image.size)
    
    # 4. Utiliser le masque nettoyé pour l'inpainting
    # ⚠️ IMPORTANT: Utiliser facade_clean, PAS facade_full
    
    mask_for_inpainting = facade_masks["facade_clean"]  # ← SANS fenêtres
    
    # 5. Appliquer l'inpainting
    # result = inpaint(
    #     image=image,
    #     mask=mask_for_inpainting,
    #     prompt="white modern facade with smooth texture"
    # )
    
    # ✅ Résultat: Façade modifiée, fenêtres intactes
    
    return mask_for_inpainting


# =====================================================
# EXEMPLE 2: Modifier uniquement le tiers supérieur
# =====================================================

def example_change_upper_facade(image_path: str):
    """
    Change uniquement le tiers supérieur de la façade
    """
    
    image = Image.open(image_path)
    semantic_map = semantic_segment(image, model_type="oneformer")
    facade_masks = prepare_facade_masks(semantic_map, image.size)
    
    # Utiliser le masque du tiers supérieur nettoyé
    mask_for_inpainting = facade_masks["facade_upper_clean"]
    
    # result = inpaint(
    #     image=image,
    #     mask=mask_for_inpainting,
    #     prompt="modern white upper facade"
    # )
    
    return mask_for_inpainting


# =====================================================
# EXEMPLE 3: Pipeline complet avec SDXL Inpainting
# =====================================================

def pipeline_with_facade_separation(
    image_path: str,
    prompt: str,
    target_zone: str = "full"  # "full", "upper", "middle", "lower"
):
    """
    Pipeline complet avec séparation façade/ouvertures
    
    Args:
        image_path: Chemin vers l'image
        prompt: Prompt pour la modification
        target_zone: Zone à modifier ("full", "upper", "middle", "lower")
    """
    
    from models.sdxl import load_sdxl_inpaint
    from steps.step3b_inpaint import generate_with_inpainting
    
    print("=" * 60)
    print("🏛️  PIPELINE AVEC SÉPARATION FAÇADE/OUVERTURES")
    print("=" * 60)
    
    # 1. Charger l'image
    print("\n📥 Chargement de l'image...")
    image = Image.open(image_path)
    
    # 2. Segmentation sémantique
    print("\n🔷 Segmentation avec OneFormer...")
    semantic_map = semantic_segment(image, model_type="oneformer")
    
    # 3. Préparer les masques
    print("\n🔧 Séparation façade/ouvertures...")
    facade_masks = prepare_facade_masks(semantic_map, image.size)
    
    # 4. Sélectionner le masque selon la zone
    mask_key = {
        "full": "facade_clean",
        "upper": "facade_upper_clean",
        "middle": "facade_middle_clean",
        "lower": "facade_lower_clean"
    }[target_zone]
    
    mask = facade_masks[mask_key]
    
    if mask is None:
        print(f"❌ Masque {mask_key} non disponible")
        return None
    
    print(f"   ✅ Masque sélectionné: {mask_key}")
    
    # 5. Charger SDXL Inpainting
    print("\n🔧 Chargement de SDXL Inpainting...")
    pipe = load_sdxl_inpaint()
    
    # 6. Génération avec inpainting
    print(f"\n🎨 Génération avec prompt: {prompt}")
    result = generate_with_inpainting(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=30,
        strength=0.8,
        guidance_scale=7.5
    )
    
    # 7. Sauvegarder
    output_path = f"output/inpaint_{target_zone}_facade.png"
    result.save(output_path)
    print(f"\n💾 Résultat sauvegardé: {output_path}")
    
    print("\n✅ Pipeline terminé!")
    print(f"\n📊 Statistiques:")
    print(f"   - Zone modifiée: {target_zone}")
    print(f"   - Fenêtres protégées: ✅")
    print(f"   - Portes protégées: ✅")
    
    return result


# =====================================================
# UTILISATION
# =====================================================

if __name__ == "__main__":
    
    # Test avec une image
    IMAGE_PATH = "input/building.jpg"
    
    # Exemple 1: Modifier toute la façade
    print("\n" + "=" * 60)
    print("EXEMPLE 1: Modifier toute la façade")
    print("=" * 60)
    
    # result = pipeline_with_facade_separation(
    #     image_path=IMAGE_PATH,
    #     prompt="modern white facade with smooth minimalist texture",
    #     target_zone="full"
    # )
    
    # Exemple 2: Modifier uniquement le tiers supérieur
    print("\n" + "=" * 60)
    print("EXEMPLE 2: Modifier le tiers supérieur")
    print("=" * 60)
    
    # result = pipeline_with_facade_separation(
    #     image_path=IMAGE_PATH,
    #     prompt="dark grey modern upper facade",
    #     target_zone="upper"
    # )
    
    # Exemple 3: Code manuel
    print("\n" + "=" * 60)
    print("EXEMPLE 3: Code manuel")
    print("=" * 60)
    
    print("""
    from segmentation import semantic_segment, prepare_facade_masks
    from PIL import Image
    
    # 1. Segmentation
    image = Image.open("building.jpg")
    semantic_map = semantic_segment(image, model_type="oneformer")
    
    # 2. Séparation façade/ouvertures
    facade_masks = prepare_facade_masks(semantic_map, image.size)
    
    # 3. Utiliser le masque
    # Option A: Toute la façade
    mask = facade_masks["facade_clean"]
    
    # Option B: Tiers supérieur seulement
    mask = facade_masks["facade_upper_clean"]
    
    # Option C: Tiers inférieur seulement
    mask = facade_masks["facade_lower_clean"]
    
    # 4. Inpainting
    result = inpaint(image, mask, "white modern facade")
    
    # ✅ Les fenêtres et portes sont automatiquement protégées!
    """)
