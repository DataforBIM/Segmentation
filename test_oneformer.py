# Test de segmentation avec OneFormer
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from segmentation.semantic_segmentation import semantic_segment, extract_architectural_masks
import os

# URL de l'image de test (façade)
IMAGE_URL = "https://res.cloudinary.com/ddmzn1508/image/upload/v1770198200/DEMO/test-project/static/Galerie/BAC_JARDIN.jpg"

print("=" * 60)
print("🔷 TEST SEGMENTATION AVEC ONEFORMER")
print("=" * 60)

# Charger l'image
print("\n📥 Chargement de l'image...")
response = requests.get(IMAGE_URL)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"   ✅ Image chargée: {image.size}")

# Segmentation sémantique avec OneFormer
print("\n🔷 Segmentation avec OneFormer...")
semantic_map = semantic_segment(image, model_type="oneformer")

# Afficher les classes détectées
print(f"\n📊 Classes détectées ({len(semantic_map.detected_classes)}):")
for i, class_name in enumerate(semantic_map.detected_classes[:20]):
    mask_array = np.array(semantic_map.masks[class_name])
    coverage = np.sum(mask_array > 0) / (image.size[0] * image.size[1])
    print(f"   {i+1}. {class_name}: {coverage*100:.1f}%")

# Extraire les masques architecturaux spécialisés
print("\n🏛️  Extraction des masques architecturaux...")
arch_masks = extract_architectural_masks(semantic_map, image.size, split_facade=True)

print(f"\n📦 Masques architecturaux extraits:")
for name, mask in arch_masks.items():
    import numpy as np
    coverage = np.sum(np.array(mask) > 0) / (mask.size[0] * mask.size[1])
    print(f"   - {name}: {coverage:.1%}")
    
    # Sauvegarder
    output_path = f"output/arch_{name}.png"
    mask.save(output_path)
    print(f"     💾 {output_path}")

print("\n✅ Test terminé!")
print("\nMasques disponibles:")
print(f"   semantic_masks = {{")
for name in arch_masks.keys():
    print(f'      "{name}": mask,')
print(f"   }}")
