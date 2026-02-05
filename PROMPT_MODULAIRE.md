# Structure de Prompt Modulaire

## Vue d'ensemble

Le système de prompts modulaires remplace l'ancien système de détection de scène par une approche configurable et flexible. Au lieu de détecter automatiquement le type de scène (INTERIOR/EXTERIOR/AERIAL), vous construisez maintenant des prompts en assemblant des modules.

## Structure des Modules

### 1. **[SCENE STRUCTURE]** - Structure et composition
- `interior`: Espace architectural intérieur
- `exterior`: Vue architecturale extérieure  
- `aerial`: Vue aérienne / bird's eye
- `landscape`: Vue paysagère architecturale
- `detail`: Détail architectural en gros plan

### 2. **[SUBJECT]** - Sujet principal
- `building`: Bâtiment contemporain
- `facade`: Façade de bâtiment
- `interior_space`: Espace intérieur architectural
- `urban_block`: Bloc urbain architectural
- `roof`: Structure de toiture
- `courtyard`: Cour intérieure architecturale
- `entrance`: Entrée de bâtiment

### 3. **[ENVIRONMENT]** - Contexte et environnement
- `urban`: Environnement urbain
- `residential`: Quartier résidentiel
- `park`: Cadre de parc
- `street`: Contexte de rue
- `plaza`: Environnement de place publique
- `isolated`: Cadre isolé / fond minimal
- `waterfront`: Cadre en bord de l'eau

### 4. **[CAMERA / LENS]** - Prise de vue et perspective
- `eye_level`: Caméra à hauteur des yeux
- `low_angle`: Prise de vue en contre-plongée
- `high_angle`: Prise de vue en plongée
- `aerial_orthogonal`: Vue aérienne orthogonale
- `aerial_oblique`: Vue aérienne oblique
- `wide_angle`: Objectif grand angle
- `normal_lens`: Objectif normal (50mm)
- `telephoto`: Téléobjectif
- `straight_verticals`: Lignes verticales droites (correction perspective)

### 5. **[LIGHTING]** - Éclairage et atmosphère
- `natural_daylight`: Lumière du jour naturelle
- `golden_hour`: Lumière dorée (sunset)
- `overcast`: Ciel couvert / lumière diffuse
- `blue_hour`: Heure bleue (twilight)
- `bright_sun`: Soleil vif
- `soft_shadows`: Ombres douces
- `hard_shadows`: Ombres dures
- `neutral_lighting`: Éclairage neutre équilibré

### 6. **[MATERIALS]** - Matériaux et textures
- `concrete`: Matériau béton
- `brick`: Matériau brique
- `glass`: Matériau verre
- `wood`: Matériau bois
- `metal`: Matériau métal
- `stone`: Matériau pierre
- `plaster`: Matériau enduit
- `mixed_materials`: Palette de matériaux variés
- `realistic_weathering`: Vieillissement réaliste
- `clean_surfaces`: Surfaces propres

### 7. **[STYLE]** - Style photographique et traitement
- `photorealistic`: Photoréaliste
- `architectural_photo`: Photographie architecturale professionnelle
- `high_quality`: Haute qualité (8k)
- `natural_colors`: Couleurs naturelles réalistes
- `minimal_processing`: Traitement minimal
- `documentary`: Style documentaire
- `clean_composition`: Composition claire

### 8. **[NEGATIVE PROMPT]** - Éléments à éviter
Géré automatiquement par le système, inclut:
- Artefacts visuels
- Rendus 3D/CGI
- Styles artistiques/cartoon
- Matériaux artificiels
- Couleurs irréalistes
- Géométrie déformée

## Utilisation

### Mode Auto-détection (Recommandé)

Le système détecte automatiquement les paramètres depuis votre prompt:

```python
from pipeline import run_pipeline

result = run_pipeline(
    image_url="https://...",
    user_prompt="modern concrete building in urban area with glass facade",
    auto_detect_prompt=True,  # Active l'auto-détection (défaut)
    enable_sdxl=True
)
```

Le système détectera automatiquement:
- `scene_structure`: "exterior" (depuis "building")
- `subject`: "building" 
- `environment`: "urban" (depuis "urban area")
- `materials`: ["concrete", "glass"] (depuis les mots-clés)
- etc.

### Mode Manuel (Contrôle Total)

Pour un contrôle précis, spécifiez manuellement les paramètres:

```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="villa avec piscine",
    
    # Configuration manuelle
    scene_structure="exterior",
    subject="building",
    environment="residential",
    camera=["eye_level", "wide_angle", "straight_verticals"],
    lighting="golden_hour",
    materials=["concrete", "glass", "wood"],
    style=["photorealistic", "architectural_photo", "high_quality"],
    
    auto_detect_prompt=False,  # Désactiver l'auto-détection
    enable_sdxl=True
)
```

### Mode Hybride (Override Partiel)

Combinez auto-détection et overrides manuels:

```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="building renovation",
    
    # Auto-détection activée, mais avec overrides
    auto_detect_prompt=True,
    lighting="golden_hour",  # Override manuel de l'éclairage
    materials=["brick", "wood"],  # Override manuel des matériaux
    
    enable_sdxl=True
)
```

## Exemples Pratiques

### Exemple 1: Bâtiment Moderne Urbain

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
    style=["photorealistic", "architectural_photo", "high_quality"],
    enable_sdxl=True
)
```

### Exemple 2: Intérieur Résidentiel

```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="modern living room with large windows",
    scene_structure="interior",
    subject="interior_space",
    environment="residential",
    camera=["eye_level", "wide_angle"],
    lighting="natural_daylight",
    materials=["wood", "concrete", "glass"],
    style=["photorealistic", "architectural_photo", "natural_colors"],
    enable_sdxl=True
)
```

### Exemple 3: Vue Aérienne Urbaine

```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="urban block aerial reconstruction",
    scene_structure="aerial",
    subject="urban_block",
    environment="urban",
    camera=["aerial_oblique"],
    lighting="overcast",
    materials=["mixed_materials", "realistic_weathering"],
    style=["photorealistic", "documentary", "high_quality"],
    enable_sdxl=True
)
```

### Exemple 4: Façade en Golden Hour

```python
result = run_pipeline(
    image_url="https://...",
    user_prompt="brick facade with wooden windows",
    scene_structure="exterior",
    subject="facade",
    environment="residential",
    camera=["eye_level", "normal_lens", "straight_verticals"],
    lighting="golden_hour",
    materials=["brick", "wood", "realistic_weathering"],
    style=["photorealistic", "architectural_photo"],
    enable_sdxl=True
)
```

## Migration depuis l'Ancien Système

### Ancien Système (Détection de Scène)

```python
# ❌ Ancienne API (déprécié)
result = run_pipeline(
    image_url="https://...",
    user_prompt="modern building",
    enable_scene_detection=True,  # Détection automatique INTERIOR/EXTERIOR/AERIAL
    enable_sdxl=True
)
```

### Nouveau Système (Prompts Modulaires)

```python
# ✅ Nouvelle API (recommandé)
result = run_pipeline(
    image_url="https://...",
    user_prompt="modern building",
    auto_detect_prompt=True,  # Auto-détection depuis le prompt
    # Ou spécifier manuellement:
    # scene_structure="exterior",
    # subject="building",
    # etc.
    enable_sdxl=True
)
```

## Avantages de la Nouvelle Approche

1. **Flexibilité**: Contrôle précis sur chaque aspect du prompt
2. **Transparence**: Plus de "boîte noire", vous voyez exactement ce qui est généré
3. **Reproductibilité**: Configurations réutilisables et documentables
4. **Extensibilité**: Facile d'ajouter de nouveaux modules sans changer la logique
5. **Auto-détection Intelligente**: Détection automatique des paramètres depuis le prompt utilisateur

## Structure des Fichiers

```
prompts/
├── modular_structure.py    # Définition de tous les modules
├── modular_builder.py      # Logique de construction des prompts
├── builders.py             # Interface principale (nouveau)
├── base.py                 # (ancien) Base prompts
├── scenes.py               # (ancien) Scene prompts
└── aerial_elements.py      # (ancien) Aerial prompts
```

## API Programmatique

### Utiliser le Builder Directement

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

### Utiliser la Configuration Objet

```python
from prompts.modular_builder import PromptConfig, build_modular_prompt

config = PromptConfig()
config.set_user_prompt("modern villa with pool")
config.set_scene_structure("exterior")
config.set_subject("building")
config.set_environment("residential")
config.set_camera(["eye_level", "wide_angle", "straight_verticals"])
config.set_lighting("golden_hour")
config.set_materials(["concrete", "glass", "wood"])
config.set_style(["photorealistic", "architectural_photo"])
config.add_custom_positive("infinity pool, luxury design")
config.add_custom_negative("old, deteriorated")

prompt, negative = build_modular_prompt(config)
```

### Auto-détection

```python
from prompts.modular_builder import auto_detect_config_from_prompt, build_modular_prompt

config = auto_detect_config_from_prompt(
    "modern concrete building with large glass windows in urban area at sunset"
)

prompt, negative = build_modular_prompt(config)
```

## Notes Techniques

- **Ordre des modules**: Le prompt utilisateur a toujours la priorité maximale
- **Paramètres multiples**: Camera, materials et style acceptent des listes
- **Compatibilité**: L'ancien système via `build_prompts_legacy()` reste disponible
- **Performance**: Aucun impact sur les performances, juste une meilleure organisation
