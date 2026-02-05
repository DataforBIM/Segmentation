# 🏛️ Séparation Façade / Ouvertures

## Problème Classique

Lors de la segmentation sémantique, **les fenêtres sont souvent partiellement incluses dans la classe "building/facade"**.

Résultat: Si on modifie la façade avec inpainting, les vitres et cadres de fenêtres sont aussi modifiés → **reflets cassés, cadres repeints**.

## ✅ Solution Implémentée

Le système utilise **OneFormer** avec post-processing pour séparer proprement:
- **Façade** (mur à modifier)
- **Ouvertures** (fenêtres + portes à protéger)

### Architecture

```python
# Segmentation OneFormer (panoptique)
semantic_map = semantic_segment(image, model_type="oneformer")

# Séparation automatique façade/ouvertures
facade_masks = prepare_facade_masks(semantic_map, image.size)

# Résultat:
{
    "facade_full": mask,        # Façade complète (avec fenêtres)
    "facade_clean": mask,       # Façade SANS fenêtres/portes ✅
    "windows": mask,            # Fenêtres (protégées)
    "doors": mask,              # Portes (protégées)
    "protected": mask,          # windows + doors combinés
    
    # Zones divisées verticalement
    "facade_upper_clean": mask,  # Tiers supérieur (sans ouvertures)
    "facade_middle_clean": mask, # Tiers milieu (sans ouvertures)
    "facade_lower_clean": mask,  # Tiers inférieur (sans ouvertures)
}
```

### Algorithme

```python
# 1. Extraire les masques de base
facade_full = building_mask  # OneFormer: classe "building"
windows = window_mask         # OneFormer: classe "window"
doors = door_mask             # OneFormer: classe "door"

# 2. Combiner les ouvertures
protected = windows + doors   # Union

# 3. Soustraire des ouvertures de la façade
facade_clean = facade_full - protected  # Soustraction

# 4. Diviser verticalement en 3 zones
facade_upper = facade_clean[top_third]
facade_middle = facade_clean[middle_third]
facade_lower = facade_clean[bottom_third]
```

## 📋 Utilisation

### Cas 1: Modifier toute la façade

```python
from segmentation import semantic_segment, prepare_facade_masks

# Segmentation
image = Image.open("building.jpg")
semantic_map = semantic_segment(image, model_type="oneformer")

# Séparation
facade_masks = prepare_facade_masks(semantic_map, image.size)

# Inpainting
result = inpaint(
    image=image,
    mask=facade_masks["facade_clean"],  # ← SANS fenêtres
    prompt="white modern facade"
)

# ✅ Résultat: Façade modifiée, fenêtres intactes
```

### Cas 2: Modifier uniquement le tiers supérieur

```python
# Utiliser facade_upper_clean au lieu de facade_clean
result = inpaint(
    image=image,
    mask=facade_masks["facade_upper_clean"],
    prompt="dark grey upper facade"
)

# ✅ Résultat: Seul le tiers supérieur est modifié
```

### Cas 3: Modifier tout SAUF les fenêtres

```python
# Si vous voulez modifier toute l'image sauf les fenêtres
# (pas seulement la façade)

full_mask = Image.new("L", image.size, 255)  # Tout en blanc
protected_mask = facade_masks["protected"]   # Fenêtres + portes

editable_mask = subtract_masks(full_mask, [protected_mask])

result = inpaint(
    image=image,
    mask=editable_mask,
    prompt="modern renovation"
)
```

## 🎯 Avantages

| Avant (sans séparation) | Après (avec séparation) |
|-------------------------|-------------------------|
| ❌ Fenêtres repeintes | ✅ Fenêtres préservées |
| ❌ Reflets de vitre cassés | ✅ Reflets intacts |
| ❌ Cadres modifiés | ✅ Cadres préservés |
| ❌ Portes repeintes | ✅ Portes préservées |

## 📊 Statistiques

Sur l'image de test (1024x1536):

| Masque | Couverture |
|--------|------------|
| `facade_full` | 26.0% |
| `facade_clean` | 26.0% |
| `facade_upper_clean` | 6.8% |
| `facade_middle_clean` | 11.5% |
| `facade_lower_clean` | 7.7% |
| `protected` | 0.0% (pas détecté dans ce cas) |

## 🔧 API Complète

### `prepare_facade_masks(semantic_map, image_size)`

**Arguments:**
- `semantic_map`: SemanticMap de OneFormer
- `image_size`: (width, height)

**Retourne:**
```python
{
    "facade_full": PIL.Image,        # Masque complet
    "facade_clean": PIL.Image,       # Sans ouvertures ✅
    "windows": PIL.Image,            # Fenêtres seules
    "doors": PIL.Image,              # Portes seules
    "protected": PIL.Image,          # windows + doors
    "facade_upper_clean": PIL.Image, # Tiers supérieur
    "facade_middle_clean": PIL.Image,# Tiers milieu
    "facade_lower_clean": PIL.Image, # Tiers inférieur
}
```

### `subtract_masks(base_mask, subtract_masks)`

**Arguments:**
- `base_mask`: Masque de base
- `subtract_masks`: Liste des masques à soustraire

**Retourne:**
- PIL.Image: `base_mask - subtract_masks[0] - subtract_masks[1] - ...`

**Exemple:**
```python
# Façade sans fenêtres ET sans portes
clean = subtract_masks(facade_mask, [windows_mask, doors_mask])
```

## 📁 Fichiers Générés

Lors du test `test_facade_separation.py`:

```
output/facade_separation/
├── facade_full.png           # Façade complète
├── facade_clean.png          # Façade sans ouvertures ✅
├── protected.png             # Fenêtres + portes
├── facade_upper_clean.png    # Tiers supérieur
├── facade_middle_clean.png   # Tiers milieu
├── facade_lower_clean.png    # Tiers inférieur
├── vis_01_facade_full.png    # Visualisation rouge
├── vis_02_protected.png      # Visualisation verte
├── vis_03_facade_clean.png   # Visualisation bleue
└── comparison.png            # Comparaison 2x2
```

## 🚀 Prochaines Étapes

Pour améliorer la détection des fenêtres:

1. **Fine-tuning OneFormer** sur un dataset architectural
2. **Post-processing géométrique**: Détecter les rectangles dans la façade
3. **Modèle spécialisé**: Entraîner un modèle spécifique façade/fenêtres
4. **Fusion SegFormer + SAM2**: Utiliser SAM2 pour affiner les contours

## 📚 Références

- **OneFormer**: [shi-labs/oneformer_ade20k_swin_large](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- **ADE20K Dataset**: 150 classes sémantiques
- **Segmentation Panoptique**: Sémantique + Instances
