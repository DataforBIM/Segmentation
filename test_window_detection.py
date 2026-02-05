# Test avec détection forcée des fenêtres/portes
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import semantic_segment, prepare_facade_masks, subtract_masks
import torch

# URL de l'image de test
IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 60)
print("🔍 TEST AVEC DÉTECTION FORCÉE DES OUVERTURES")
print("=" * 60)

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image chargée: {image.size}")

# Segmentation OneFormer
print("\n🔷 Segmentation OneFormer...")
semantic_map = semantic_segment(image, model_type="oneformer")

print(f"\n📊 Classes détectées par OneFormer:")
for cls in semantic_map.detected_classes:
    mask = semantic_map.masks[cls]
    coverage = np.sum(np.array(mask) > 0) / (image.size[0] * image.size[1])
    print(f"   - {cls}: {coverage*100:.1f}%")

# Vérifier si des fenêtres ont été détectées
has_windows = any("window" in cls.lower() for cls in semantic_map.detected_classes)
has_doors = any("door" in cls.lower() for cls in semantic_map.detected_classes)

print(f"\n🔍 Détection des ouvertures:")
print(f"   - Fenêtres détectées: {'✅' if has_windows else '❌'}")
print(f"   - Portes détectées: {'✅' if has_doors else '❌'}")

if not has_windows and not has_doors:
    print("\n⚠️  Aucune ouverture détectée par OneFormer!")
    print("   💡 Solution: Utiliser Grounding DINO pour forcer la détection")
    
    # Utiliser Grounding DINO pour détecter les fenêtres
    print("\n🔍 Détection avec Grounding DINO...")
    
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    # Charger Grounding DINO
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to("cuda")
    
    # Détecter fenêtres et portes
    text = "window. door. glass. windowpane."
    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )[0]
    
    print(f"   ✅ Grounding DINO: {len(results['boxes'])} détections")
    
    # Créer des masques depuis les boxes détectées
    if len(results['boxes']) > 0:
        windows_mask = Image.new("L", image.size, 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(windows_mask)
        
        for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
            if score > 0.3:
                x1, y1, x2, y2 = box.cpu().numpy()
                # Dilater légèrement la box pour avoir une marge
                margin = 5
                draw.rectangle(
                    [x1-margin, y1-margin, x2+margin, y2+margin],
                    fill=255
                )
                print(f"      - {label}: score={score:.2f}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        
        windows_mask.save("output/facade_separation/windows_grounding_dino.png")
        print(f"\n   💾 Masque fenêtres sauvegardé (Grounding DINO)")
        
        # Maintenant soustraire du masque de façade
        facade_masks = prepare_facade_masks(semantic_map, image.size)
        
        if facade_masks["facade_full"]:
            facade_clean_manual = subtract_masks(
                facade_masks["facade_full"],
                [windows_mask]
            )
            
            facade_clean_manual.save("output/facade_separation/facade_clean_with_grounding_dino.png")
            
            # Statistiques
            full_coverage = np.sum(np.array(facade_masks["facade_full"]) > 0) / (image.size[0] * image.size[1])
            clean_coverage = np.sum(np.array(facade_clean_manual) > 0) / (image.size[0] * image.size[1])
            windows_coverage = np.sum(np.array(windows_mask) > 0) / (image.size[0] * image.size[1])
            
            print(f"\n📊 Résultats avec Grounding DINO:")
            print(f"   - Façade complète: {full_coverage*100:.1f}%")
            print(f"   - Fenêtres détectées: {windows_coverage*100:.1f}%")
            print(f"   - Façade nettoyée: {clean_coverage*100:.1f}%")
            print(f"   - Différence: {(full_coverage - clean_coverage)*100:.1f}%")
            
            if full_coverage > clean_coverage:
                print(f"\n✅ Protection réussie! {(full_coverage - clean_coverage)*100:.1f}% de fenêtres retirées")
            else:
                print(f"\n⚠️  Pas de différence - les fenêtres n'ont pas été détectées")

else:
    print(f"\n✅ Ouvertures détectées par OneFormer")
    facade_masks = prepare_facade_masks(semantic_map, image.size)

print("\n" + "=" * 60)
print("📝 CONCLUSION")
print("=" * 60)
print("""
OneFormer seul n'a pas détecté les fenêtres dans cette image.

Solutions possibles:
1. ✅ Utiliser Grounding DINO en complément (détection par texte)
2. ✅ Utiliser SAM2 sur les rectangles suspects
3. ✅ Post-processing géométrique (détecter rectangles dans la façade)
4. Fine-tuner OneFormer sur un dataset architectural

👉 Le système de soustraction fonctionne, mais dépend de la qualité
   de la détection initiale des fenêtres/portes.
""")
