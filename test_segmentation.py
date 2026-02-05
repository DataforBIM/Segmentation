# =====================================================
# TEST SEGMENTATION PIPELINE
# =====================================================
# Script pour tester le nouveau pipeline de segmentation

from PIL import Image
import os

# Import du nouveau système de segmentation
from segmentation import (
    # Pipeline principal
    segment_from_prompt,
    quick_segment,
    segment_element,
    load_segmentation_models,
    
    # Étapes individuelles
    parse_intent,
    resolve_target,
    semantic_segment,
    fuse_masks,
    refine_mask,
    validate_mask
)


def test_intent_parser():
    """Test l'intent parser avec différents prompts"""
    
    print("=" * 60)
    print("TEST: INTENT PARSER")
    print("=" * 60)
    
    test_prompts = [
        "change the floor to marble",
        "replace wall with brick texture",
        "add wooden flooring",
        "make the ceiling white",
        "change furniture to modern style",
        "replace rug with persian carpet",
        "change lighting to warm tone"
    ]
    
    for prompt in test_prompts:
        intent = parse_intent(prompt)
        print(f"\n📝 Prompt: \"{prompt}\"")
        print(f"   Action: {intent.action}")
        print(f"   Target: {intent.target_hint}")
        print(f"   Material: {intent.material}")
        print(f"   Color: {intent.color}")
        print(f"   Style: {intent.style}")


def test_target_resolver():
    """Test la résolution des cibles"""
    
    print("\n" + "=" * 60)
    print("TEST: TARGET RESOLVER")
    print("=" * 60)
    
    test_prompts = [
        "change the floor to marble",
        "replace the walls",
        "change furniture style",
        "modify the lighting"
    ]
    
    for prompt in test_prompts:
        intent = parse_intent(prompt)
        target = resolve_target(intent)
        
        print(f"\n📝 Prompt: \"{prompt}\"")
        print(f"   Primary: {target.primary}")
        print(f"   Protected: {target.protected}")
        print(f"   Context: {target.context}")
        print(f"   Method: {target.method}")


def test_full_pipeline(image_path: str):
    """Test le pipeline complet"""
    
    print("\n" + "=" * 60)
    print("TEST: FULL PIPELINE")
    print("=" * 60)
    
    # Charger l'image
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    print(f"📷 Image chargée: {image.size}")
    
    # Charger les modèles
    models = load_segmentation_models(device="cuda")
    
    # Test avec différents prompts
    test_prompts = [
        "change the floor to white marble",
        "replace walls with modern texture"
    ]
    
    for prompt in test_prompts:
        print(f"\n🎯 Test: \"{prompt}\"")
        
        result = segment_from_prompt(
            image=image,
            user_prompt=prompt,
            sam2_predictor=models.get("sam2_predictor"),
            segformer_model=models.get("segformer_model"),
            segformer_processor=models.get("segformer_processor"),
            device="cuda",
            verbose=True
        )
        
        # Sauvegarder le masque
        output_name = prompt.replace(" ", "_")[:30] + "_mask.png"
        output_path = os.path.join("output", output_name)
        result.final_mask.save(output_path)
        print(f"💾 Masque sauvegardé: {output_path}")


def test_quick_segment(image_path: str):
    """Test la segmentation rapide"""
    
    print("\n" + "=" * 60)
    print("TEST: QUICK SEGMENT")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    
    # Segmentation rapide du sol
    mask = quick_segment(
        image=image,
        target_classes=["floor", "rug", "carpet"],
        protected_classes=["person", "furniture"]
    )
    
    # Sauvegarder
    mask.save("output/quick_segment_floor.png")
    print("✅ Masque sauvegardé: output/quick_segment_floor.png")


def test_element_segment(image_path: str):
    """Test la segmentation par élément"""
    
    print("\n" + "=" * 60)
    print("TEST: ELEMENT SEGMENT")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    
    elements = ["floor", "wall", "ceiling", "furniture", "window"]
    
    for element in elements:
        try:
            mask = segment_element(image, element)
            output_path = f"output/element_{element}_mask.png"
            mask.save(output_path)
            print(f"✅ {element}: {output_path}")
        except Exception as e:
            print(f"❌ {element}: {e}")


# =====================================================
# EXAMPLES D'UTILISATION
# =====================================================

def example_basic_usage():
    """Exemple d'utilisation basique"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  EXEMPLES D'UTILISATION DU PIPELINE DE SEGMENTATION      ║
    ╚══════════════════════════════════════════════════════════╝
    
    # 1. UTILISATION COMPLÈTE AVEC PROMPT
    # ------------------------------------
    
    from segmentation import segment_from_prompt, load_segmentation_models
    from PIL import Image
    
    # Charger les modèles une fois
    models = load_segmentation_models()
    
    # Charger une image
    image = Image.open("input/room.jpg")
    
    # Segmenter avec un prompt naturel
    result = segment_from_prompt(
        image=image,
        user_prompt="change the floor to marble",
        **models
    )
    
    # Utiliser le masque résultant
    mask = result.final_mask
    print(f"Coverage: {result.coverage:.1%}")
    
    
    # 2. SEGMENTATION RAPIDE
    # -----------------------
    
    from segmentation import quick_segment
    
    mask = quick_segment(
        image=image,
        target_classes=["floor", "rug"],
        protected_classes=["person", "furniture"]
    )
    
    
    # 3. SEGMENTATION PAR ÉLÉMENT
    # ----------------------------
    
    from segmentation import segment_element
    
    floor_mask = segment_element(image, "floor")
    wall_mask = segment_element(image, "wall")
    ceiling_mask = segment_element(image, "ceiling")
    
    
    # 4. UTILISATION ÉTAPE PAR ÉTAPE
    # -------------------------------
    
    from segmentation import (
        parse_intent,
        resolve_target,
        semantic_segment,
        fuse_masks,
        refine_mask,
        validate_mask
    )
    
    # Étape 1: Parser l'intention
    intent = parse_intent("change the floor to wood")
    print(f"Target: {intent.target}, Material: {intent.material}")
    
    # Étape 2: Résoudre les cibles
    target = resolve_target(intent)
    print(f"Classes: {target.primary_classes}")
    
    # Étape 3: Segmentation sémantique
    sem_map = semantic_segment(image)
    
    # Étape 4: Fusion des masques
    layers = fuse_masks(target_mask, protected_mask)
    
    # Étape 5: Raffinement
    refined = refine_mask(layers.final_mask, dilate=3, feather=6)
    
    # Étape 6: Validation
    result = validate_mask(refined)
    if result.is_valid:
        print("✅ Masque validé!")
    """)


if __name__ == "__main__":
    
    # Afficher les exemples
    example_basic_usage()
    
    # Tests unitaires
    test_intent_parser()
    test_target_resolver()
    
    # Tests avec image (décommentez si vous avez une image)
    # test_full_pipeline("input/test_room.jpg")
    # test_quick_segment("input/test_room.jpg")
    # test_element_segment("input/test_room.jpg")
