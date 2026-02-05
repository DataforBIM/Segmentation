# 🎯 SEGMENTATION PIPELINE - Architecture ChatGPT-like

## Vue d'ensemble

Ce pipeline de segmentation intelligent analyse les prompts utilisateur en langage naturel et génère automatiquement des masques de segmentation précis pour la génération d'images.

## 📊 Architecture en 7 Étapes

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER PROMPT                                 │
│            "change the floor to white marble"                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 1: INTENT PARSER                                          │
│  ─────────────────────                                           │
│  Analyse le prompt pour extraire:                                │
│  • Action: "change"                                              │
│  • Target: "floor"                                               │
│  • Material: "marble"                                            │
│  • Color: "white"                                                │
│  • Style: null                                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 2: TARGET RESOLVER                                        │
│  ────────────────────────                                        │
│  Détermine les classes ADE20K:                                   │
│  • Primary: ["floor", "rug", "carpet"]    ← À modifier           │
│  • Protected: ["person", "furniture"]     ← Ne pas toucher       │
│  • Context: ["wall", "ceiling"]           ← Garder cohérent      │
│  • Priority: "high"                                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 3: SEMANTIC SEGMENTATION (SegFormer)                      │
│  ──────────────────────────────────────────                      │
│  • Modèle: nvidia/segformer-b5-finetuned-ade-640-640            │
│  • 150 classes ADE20K                                            │
│  • Génère une carte sémantique complète                          │
│  • Crée les masques par classe                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 4: INSTANCE SEGMENTATION (SAM2)                           │
│  ─────────────────────────────────────                           │
│  • Modèle: facebook/sam2-hiera-large                            │
│  • Affine le masque sémantique                                   │
│  • Points samples depuis le masque sémantique                    │
│  • Bords plus précis au pixel                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 5: MASK FUSION                                            │
│  ───────────────────                                             │
│  Hiérarchie des priorités:                                       │
│  1. PROTECTED (priorité max) - Jamais modifié                    │
│  2. TARGET (zone à modifier)                                     │
│  3. CONTEXT (reste de l'image)                                   │
│                                                                  │
│  Target - Protected = Final Mask                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 6: MASK REFINEMENT                                        │
│  ───────────────────────                                         │
│  Opérations morphologiques:                                      │
│  • Clean: Supprime les petites régions                           │
│  • Fill holes: Remplit les trous                                 │
│  • Smooth: Lisse les contours                                    │
│  • Dilate: Agrandit légèrement (2-4px)                          │
│  • Feather: Bords doux pour transitions (4-12px)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ÉTAPE 7: VALIDATION                                             │
│  ──────────────────                                              │
│  Critères de validation:                                         │
│  • Coverage: 5% < mask < 60%                                     │
│  • Non vide, non plein                                           │
│  • Pas trop fragmenté                                            │
│                                                                  │
│  Auto-correction si échec:                                       │
│  • Too small → Dilate ou semantic only                          │
│  • Too large → Erode ou add protection                          │
│  • Empty → Fallback semantic mask                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL MASK                                   │
│                                                                  │
│   ┌─────────────────────────────────────┐                       │
│   │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  → Protected          │
│   │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │                        │
│   │  ███████████████████████████████████│  → Target (blanc)     │
│   │  ███████████████████████████████████│                        │
│   └─────────────────────────────────────┘                       │
│                                                                  │
│   Coverage: 35% | Valid: ✅ | Time: 1.2s                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure des Fichiers

```
segmentation/
├── __init__.py              # Exports publics
├── intent_parser.py         # ÉTAPE 1: Parse les prompts
├── target_resolver.py       # ÉTAPE 2: Résout les cibles
├── semantic_segmentation.py # ÉTAPE 3: SegFormer
├── instance_segmentation.py # ÉTAPE 4: SAM2
├── mask_fusion.py           # ÉTAPE 5: Fusion hiérarchique
├── mask_refinement.py       # ÉTAPE 6: Raffinement
├── validation.py            # ÉTAPE 7: Validation
└── pipeline.py              # Orchestrateur principal
```

---

## 🚀 Utilisation Rapide

### Pipeline Complet

```python
from segmentation import segment_from_prompt, load_segmentation_models
from PIL import Image

# Charger les modèles (une fois)
models = load_segmentation_models()

# Charger l'image
image = Image.open("room.jpg")

# Segmenter avec un prompt naturel
result = segment_from_prompt(
    image=image,
    user_prompt="change the floor to marble",
    **models
)

# Utiliser le masque
mask = result.final_mask
print(f"Coverage: {result.coverage:.1%}")
```

### Segmentation Rapide

```python
from segmentation import quick_segment

mask = quick_segment(
    image=image,
    target_classes=["floor", "rug"],
    protected_classes=["person", "furniture"]
)
```

### Segmentation par Élément

```python
from segmentation import segment_element

floor_mask = segment_element(image, "floor")
wall_mask = segment_element(image, "wall")
```

---

## 🔧 Classes ADE20K Supportées

| Élément | Classes ADE20K |
|---------|----------------|
| Floor | floor, rug, carpet, mat |
| Wall | wall |
| Ceiling | ceiling |
| Furniture | sofa, chair, table, bed, cabinet |
| Window | window, windowpane |
| Door | door |
| Light | lamp, chandelier, light |
| Plant | plant, tree, flower |

---

## 📊 Paramètres de Raffinement Dynamiques

Les paramètres s'adaptent automatiquement selon:

| Image Size | Dilate | Feather | Min Area |
|------------|--------|---------|----------|
| < 512px    | 1px    | 2px     | 50px²    |
| < 1024px   | 2px    | 4px     | 100px²   |
| < 2048px   | 3px    | 6px     | 200px²   |
| ≥ 2048px   | 4px    | 8px     | 400px²   |

Et selon la couverture:
- **< 10%**: Feather réduit (garder précision)
- **> 50%**: Feather augmenté (transitions douces)

---

## ✅ Validation et Fallback

### Critères de Validation

| Critère | Seuil | Action si Échec |
|---------|-------|-----------------|
| Coverage min | 5% | Dilate ou semantic only |
| Coverage max | 60% | Erode ou add protection |
| Empty | < 1% | Fallback to default mask |
| Full | > 95% | Add more protection |
| Fragments | > 10 | Clean small regions |

### Stratégies de Fallback

1. **Dilate** - Agrandit le masque
2. **Erode** - Réduit le masque
3. **Semantic Only** - Utilise uniquement SegFormer
4. **Clean Fragments** - Supprime les petites régions
5. **Default Mask** - Masque elliptique au centre

---

## 🎨 Exemples de Prompts Supportés

```
✓ "change the floor to marble"
✓ "replace wall with brick texture"
✓ "make the ceiling white"
✓ "add wooden flooring"
✓ "change furniture to modern style"
✓ "replace rug with persian carpet"
✓ "modify lighting to warm tone"
```

---

## ⚡ Performance

- **SegFormer**: ~0.5s sur GPU
- **SAM2**: ~0.3s par instance
- **Total Pipeline**: ~1-2s pour une image 1024x1024

---

## 📝 Notes Importantes

1. **GPU Requis**: CUDA recommandé pour les performances
2. **Mémoire**: ~4GB VRAM minimum
3. **Modèles**: Téléchargés automatiquement au premier lancement
4. **Scipy**: Requis pour les opérations morphologiques
