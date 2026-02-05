# Test pour voir TOUTES les classes détectées par OneFormer
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import semantic_segment, subtract_masks
import os

IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 60)
print("🔍 INSPECTION COMPLÈTE DES CLASSES ONEFORMER")
print("=" * 60)

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image chargée: {image.size}")

# Segmentation OneFormer
print("\n🔷 Segmentation OneFormer...")
semantic_map = semantic_segment(image, model_type="oneformer")

# Afficher TOUTES les classes détectées
print(f"\n📊 Classes détectées par OneFormer ({len(semantic_map.detected_classes)}):")
print("=" * 60)

os.makedirs("output/oneformer_classes", exist_ok=True)

for i, class_name in enumerate(semantic_map.detected_classes):
    mask = semantic_map.masks[class_name]
    mask_array = np.array(mask)
    coverage = np.sum(mask_array > 0) / (image.size[0] * image.size[1])
    
    print(f"{i+1:2d}. {class_name:20s} : {coverage*100:6.2f}%")
    
    # Sauvegarder chaque masque
    output_path = f"output/oneformer_classes/{i+1:02d}_{class_name}.png"
    mask.save(output_path)

# Chercher spécifiquement les fenêtres et portes
print("\n" + "=" * 60)
print("🔍 Recherche de fenêtres et portes:")
print("=" * 60)

window_keywords = ["window", "windowpane", "glass", "pane"]
door_keywords = ["door", "entrance"]

windows_found = []
doors_found = []

for class_name in semantic_map.detected_classes:
    class_lower = class_name.lower()
    if any(kw in class_lower for kw in window_keywords):
        windows_found.append(class_name)
    if any(kw in class_lower for kw in door_keywords):
        doors_found.append(class_name)

print(f"\nFenêtres trouvées: {windows_found if windows_found else '❌ Aucune'}")
print(f"Portes trouvées: {doors_found if doors_found else '❌ Aucune'}")

# Tester l'équation demandée
print("\n" + "=" * 60)
print("🧪 TEST DE L'ÉQUATION:")
print("=" * 60)
print("facade_mask = facade_upper + facade_lower")
print("protected_mask = windows + doors")
print("facade_clean = facade_mask - protected_mask")
print("=" * 60)

# Extraire facade_upper et facade_lower depuis building
building_mask = semantic_map.masks.get("building")

if building_mask:
    # Diviser building verticalement
    mask_array = np.array(building_mask)
    height, width = mask_array.shape
    
    # Trouver les limites du building
    rows_with_building = np.any(mask_array > 0, axis=1)
    if np.any(rows_with_building):
        top = np.argmax(rows_with_building)
        bottom = height - np.argmax(rows_with_building[::-1])
        building_height = bottom - top
        
        # Diviser en 3 tiers
        third_height = building_height // 3
        
        # Créer facade_upper (tiers supérieur)
        facade_upper = np.zeros_like(mask_array)
        facade_upper[top:top+third_height, :] = mask_array[top:top+third_height, :]
        facade_upper_img = Image.fromarray(facade_upper, mode="L")
        facade_upper_img.save("output/oneformer_classes/facade_upper.png")
        
        # Créer facade_lower (tiers inférieur)
        facade_lower = np.zeros_like(mask_array)
        facade_lower[bottom-third_height:bottom, :] = mask_array[bottom-third_height:bottom, :]
        facade_lower_img = Image.fromarray(facade_lower, mode="L")
        facade_lower_img.save("output/oneformer_classes/facade_lower.png")
        
        # 1. facade_mask = facade_upper + facade_lower
        facade_mask_array = np.maximum(facade_upper, facade_lower)
        facade_mask = Image.fromarray(facade_mask_array, mode="L")
        facade_mask.save("output/oneformer_classes/facade_mask_combined.png")
        
        upper_coverage = np.sum(facade_upper > 0) / (width * height)
        lower_coverage = np.sum(facade_lower > 0) / (width * height)
        combined_coverage = np.sum(facade_mask_array > 0) / (width * height)
        
        print(f"\n✅ Étape 1: Combiner upper + lower")
        print(f"   facade_upper: {upper_coverage*100:.2f}%")
        print(f"   facade_lower: {lower_coverage*100:.2f}%")
        print(f"   facade_mask (upper + lower): {combined_coverage*100:.2f}%")
        
        # 2. protected_mask = windows + doors
        if windows_found or doors_found:
            protected_masks = []
            
            for class_name in windows_found + doors_found:
                protected_masks.append(np.array(semantic_map.masks[class_name]))
            
            if protected_masks:
                protected_mask_array = np.maximum.reduce(protected_masks)
                protected_mask = Image.fromarray(protected_mask_array, mode="L")
                protected_mask.save("output/oneformer_classes/protected_mask.png")
                
                protected_coverage = np.sum(protected_mask_array > 0) / (width * height)
                print(f"\n✅ Étape 2: Créer protected_mask")
                print(f"   Classes protégées: {windows_found + doors_found}")
                print(f"   protected_mask: {protected_coverage*100:.2f}%")
                
                # 3. facade_clean = facade_mask - protected_mask
                facade_clean = subtract_masks(facade_mask, [protected_mask])
                facade_clean.save("output/oneformer_classes/facade_clean_final.png")
                
                clean_coverage = np.sum(np.array(facade_clean) > 0) / (width * height)
                
                print(f"\n✅ Étape 3: Soustraire protected")
                print(f"   facade_clean: {clean_coverage*100:.2f}%")
                print(f"   Différence: {(combined_coverage - clean_coverage)*100:.2f}%")
                
                if combined_coverage > clean_coverage:
                    print(f"\n✅ SUCCÈS: {(combined_coverage - clean_coverage)*100:.2f}% de fenêtres/portes retirées!")
                else:
                    print(f"\n⚠️  Pas de différence")
        else:
            print(f"\n❌ Étape 2: Impossible - Aucune fenêtre/porte détectée")
            print(f"\n💡 EXPLICATION:")
            print(f"   OneFormer n'a détecté aucune classe 'window' ou 'door'")
            print(f"   dans cette image.")
            print(f"\n   Classes détectées: {semantic_map.detected_classes}")
            print(f"\n   Raisons possibles:")
            print(f"   - Les fenêtres sont trop petites")
            print(f"   - Les fenêtres sont classées comme 'building'")
            print(f"   - L'image n'a pas de fenêtres visibles")
            print(f"   - OneFormer ADE20K n'est pas optimisé pour l'architecture")

else:
    print("\n❌ Aucune classe 'building' détectée")

print("\n" + "=" * 60)
print("📁 Fichiers générés dans output/oneformer_classes/")
print("=" * 60)
print("   - Tous les masques de classes individuels")
print("   - facade_upper.png")
print("   - facade_lower.png")
print("   - facade_mask_combined.png (upper + lower)")
if windows_found or doors_found:
    print("   - protected_mask.png (windows + doors)")
    print("   - facade_clean_final.png (facade - protected)")
