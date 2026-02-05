# 🚀 Démarrage Rapide - Système de Prompts Modulaires

## Installation

Aucune installation supplémentaire nécessaire. Tous les fichiers sont déjà en place.

## Utilisation Basique

### 1. Mode le Plus Simple (Auto-Détection)

```python
from pipeline import run_pipeline

result = run_pipeline(
    image_url="https://res.cloudinary.com/your-image.jpg",
    user_prompt="modern concrete building with glass facade",
    auto_detect_prompt=True,  # Le système détecte tout automatiquement
    enable_sdxl=True
)

# L'image générée est dans result["image"]
result["image"].save("output/result.png")
```

### 2. Mode Manuel (Contrôle Total)

```python
result = run_pipeline(
    image_url="https://res.cloudinary.com/your-image.jpg",
    user_prompt="villa de luxe",
    
    # Spécifier tous les paramètres
    scene_structure="exterior",
    subject="building",
    environment="residential",
    camera=["eye_level", "wide_angle"],
    lighting="golden_hour",
    materials=["concrete", "glass"],
    style=["photorealistic", "high_quality"],
    
    auto_detect_prompt=False,
    enable_sdxl=True
)
```

### 3. Mode Hybride (Auto + Overrides)

```python
result = run_pipeline(
    image_url="https://res.cloudinary.com/your-image.jpg",
    user_prompt="building renovation",
    
    # Auto-détection + quelques overrides
    auto_detect_prompt=True,
    lighting="golden_hour",  # Override l'éclairage
    materials=["brick", "wood"],  # Override les matériaux
    
    enable_sdxl=True
)
```

## Paramètres Disponibles

### Scene Structure (Structure de scène)
- `interior` - Intérieur
- `exterior` - Extérieur
- `aerial` - Vue aérienne
- `landscape` - Paysage architectural
- `detail` - Détail en gros plan

### Subject (Sujet principal)
- `building` - Bâtiment
- `facade` - Façade
- `interior_space` - Espace intérieur
- `urban_block` - Bloc urbain
- `roof` - Toiture
- `courtyard` - Cour
- `entrance` - Entrée

### Environment (Environnement)
- `urban` - Urbain
- `residential` - Résidentiel
- `park` - Parc
- `street` - Rue
- `plaza` - Place
- `isolated` - Isolé
- `waterfront` - Bord de l'eau

### Camera (Caméra/Objectif)
Liste, peut contenir plusieurs valeurs:
- `eye_level` - Hauteur des yeux
- `low_angle` - Contre-plongée
- `high_angle` - Plongée
- `aerial_orthogonal` - Aérien orthogonal
- `aerial_oblique` - Aérien oblique
- `wide_angle` - Grand angle
- `normal_lens` - Objectif normal
- `telephoto` - Téléobjectif
- `straight_verticals` - Verticales droites

### Lighting (Éclairage)
- `natural_daylight` - Lumière du jour
- `golden_hour` - Heure dorée
- `overcast` - Ciel couvert
- `blue_hour` - Heure bleue
- `bright_sun` - Soleil vif
- `soft_shadows` - Ombres douces
- `hard_shadows` - Ombres dures
- `neutral_lighting` - Éclairage neutre

### Materials (Matériaux)
Liste, peut contenir plusieurs valeurs:
- `concrete` - Béton
- `brick` - Brique
- `glass` - Verre
- `wood` - Bois
- `metal` - Métal
- `stone` - Pierre
- `plaster` - Enduit
- `mixed_materials` - Matériaux mixtes
- `realistic_weathering` - Vieillissement réaliste
- `clean_surfaces` - Surfaces propres

### Style (Style photographique)
Liste, peut contenir plusieurs valeurs:
- `photorealistic` - Photoréaliste
- `architectural_photo` - Photo architecturale
- `high_quality` - Haute qualité (8k)
- `natural_colors` - Couleurs naturelles
- `minimal_processing` - Traitement minimal
- `documentary` - Documentaire
- `clean_composition` - Composition claire

## Exemples Complets

### Exemple 1: Bâtiment Moderne
```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="contemporary glass tower",
    scene_structure="exterior",
    subject="building",
    environment="urban",
    camera=["low_angle", "wide_angle", "straight_verticals"],
    lighting="bright_sun",
    materials=["glass", "metal", "concrete"],
    style=["photorealistic", "architectural_photo"],
    enable_sdxl=True
)
```

### Exemple 2: Intérieur Résidentiel
```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="modern living room",
    scene_structure="interior",
    subject="interior_space",
    environment="residential",
    camera=["eye_level", "wide_angle"],
    lighting="natural_daylight",
    materials=["wood", "concrete"],
    style=["photorealistic", "natural_colors"],
    enable_sdxl=True
)
```

### Exemple 3: Vue Aérienne
```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="urban block reconstruction",
    scene_structure="aerial",
    subject="urban_block",
    environment="urban",
    camera=["aerial_oblique"],
    lighting="overcast",
    materials=["mixed_materials"],
    style=["photorealistic", "documentary"],
    enable_sdxl=True,
    enable_segmentation=True  # Important pour aérien
)
```

## Tester Sans Exécuter le Pipeline

Pour tester la construction de prompts sans générer d'images:

```python
from prompts.modular_builder import build_prompt_from_dict

prompt, negative = build_prompt_from_dict(
    user_prompt="modern villa",
    scene_structure="exterior",
    subject="building",
    environment="residential",
    camera=["eye_level", "wide_angle"],
    lighting="golden_hour",
    materials=["concrete", "glass"],
    style=["photorealistic", "high_quality"]
)

print(f"Prompt: {prompt}")
print(f"Negative: {negative}")
```

## Fichier d'Exemples

Exécutez le fichier d'exemples pour voir tous les cas d'usage:

```bash
python examples_modular_prompts.py
```

## Documentation Complète

Consultez [PROMPT_MODULAIRE.md](PROMPT_MODULAIRE.md) pour la documentation complète.

## Support

En cas de problème:
1. Vérifiez que tous les fichiers sont à jour
2. Consultez [CHANGEMENTS.md](CHANGEMENTS.md) pour les détails de migration
3. Exécutez les tests: `python examples_modular_prompts.py`

## Résumé des Changements

**AVANT (Ancien Système)**:
- Détection automatique du type de scène (INTERIOR/EXTERIOR/AERIAL)
- Pas de contrôle sur les détails du prompt
- "Boîte noire"

**APRÈS (Nouveau Système)**:
- Configuration modulaire complète
- Auto-détection intelligente OU contrôle manuel
- Transparence totale
- Flexibilité maximale

**Migration Rapide**:
```python
# Ancien
result = run_pipeline(image_url="...", user_prompt="...", enable_scene_detection=True)

# Nouveau
result = run_pipeline(image_url="...", user_prompt="...", auto_detect_prompt=True)
```

C'est tout! Le système est opérationnel. 🎉
