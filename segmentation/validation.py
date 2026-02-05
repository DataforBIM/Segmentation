# =====================================================
# ÉTAPE 7: VALIDATION AUTOMATIQUE
# =====================================================
# Vérifie et valide le masque final
# Applique des stratégies de fallback si nécessaire

import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Callable
from enum import Enum


class ValidationStatus(Enum):
    """Statuts de validation possibles"""
    VALID = "valid"
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"
    EMPTY = "empty"
    FULL = "full"
    FRAGMENTED = "fragmented"
    INVALID = "invalid"


@dataclass
class ValidationResult:
    """Résultat de validation"""
    status: ValidationStatus
    coverage: float
    message: str
    mask: Image.Image
    is_valid: bool
    suggestions: List[str]


# =====================================================
# SEUILS DE VALIDATION
# =====================================================

# Couverture minimale (5% de l'image)
MIN_COVERAGE = 0.05

# Couverture maximale (60% de l'image)
MAX_COVERAGE = 0.60

# Seuil pour masque vide
EMPTY_THRESHOLD = 0.01

# Seuil pour masque plein
FULL_THRESHOLD = 0.95

# Nombre maximum de fragments acceptés
MAX_FRAGMENTS = 10


def validate_mask(
    mask: Image.Image,
    min_coverage: float = MIN_COVERAGE,
    max_coverage: float = MAX_COVERAGE,
    check_fragmentation: bool = True
) -> ValidationResult:
    """
    Valide un masque de segmentation
    
    Args:
        mask: Masque à valider
        min_coverage: Couverture minimale requise
        max_coverage: Couverture maximale autorisée
        check_fragmentation: Vérifier la fragmentation
    
    Returns:
        ValidationResult avec statut et suggestions
    """
    
    print(f"   🔍 Validation du masque...")
    
    mask_array = np.array(mask)
    total_pixels = mask_array.size
    white_pixels = np.sum(mask_array > 127)
    coverage = white_pixels / total_pixels
    
    print(f"      📊 Couverture: {coverage:.1%}")
    
    # Cas 1: Masque vide
    if coverage < EMPTY_THRESHOLD:
        return ValidationResult(
            status=ValidationStatus.EMPTY,
            coverage=coverage,
            message="Masque vide - aucune zone détectée",
            mask=mask,
            is_valid=False,
            suggestions=[
                "Vérifier les classes ciblées",
                "Utiliser un fallback sémantique",
                "Réduire le seuil de détection"
            ]
        )
    
    # Cas 2: Masque plein
    if coverage > FULL_THRESHOLD:
        return ValidationResult(
            status=ValidationStatus.FULL,
            coverage=coverage,
            message="Masque couvre toute l'image",
            mask=mask,
            is_valid=False,
            suggestions=[
                "Vérifier les classes protégées",
                "Augmenter la protection",
                "Réduire les classes ciblées"
            ]
        )
    
    # Cas 3: Couverture trop faible
    if coverage < min_coverage:
        return ValidationResult(
            status=ValidationStatus.TOO_SMALL,
            coverage=coverage,
            message=f"Couverture insuffisante ({coverage:.1%} < {min_coverage:.1%})",
            mask=mask,
            is_valid=False,
            suggestions=[
                "Ajouter des classes ciblées",
                "Dilater le masque",
                "Utiliser le masque sémantique complet"
            ]
        )
    
    # Cas 4: Couverture trop grande
    if coverage > max_coverage:
        return ValidationResult(
            status=ValidationStatus.TOO_LARGE,
            coverage=coverage,
            message=f"Couverture excessive ({coverage:.1%} > {max_coverage:.1%})",
            mask=mask,
            is_valid=False,
            suggestions=[
                "Réduire les classes ciblées",
                "Ajouter des protections",
                "Éroder le masque"
            ]
        )
    
    # Cas 5: Fragmentation excessive
    if check_fragmentation:
        fragments = count_mask_fragments(mask)
        if fragments > MAX_FRAGMENTS:
            return ValidationResult(
                status=ValidationStatus.FRAGMENTED,
                coverage=coverage,
                message=f"Masque trop fragmenté ({fragments} fragments)",
                mask=mask,
                is_valid=False,
                suggestions=[
                    "Nettoyer les petites régions",
                    "Augmenter le seuil minimum d'aire",
                    "Utiliser un masque plus simple"
                ]
            )
    
    # Cas 6: Tout est OK!
    print(f"   ✅ Masque validé ({coverage:.1%} couverture)")
    
    return ValidationResult(
        status=ValidationStatus.VALID,
        coverage=coverage,
        message=f"Masque valide ({coverage:.1%} couverture)",
        mask=mask,
        is_valid=True,
        suggestions=[]
    )


def count_mask_fragments(mask: Image.Image) -> int:
    """Compte le nombre de régions distinctes dans le masque"""
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    labeled, num_features = ndimage.label(mask_array)
    
    return num_features


# =====================================================
# STRATÉGIES DE FALLBACK
# =====================================================

@dataclass
class FallbackStrategy:
    """Stratégie de fallback"""
    name: str
    description: str
    action: Callable
    priority: int


def create_default_fallback_strategies() -> List[FallbackStrategy]:
    """Crée les stratégies de fallback par défaut"""
    
    return [
        FallbackStrategy(
            name="dilate",
            description="Dilater le masque pour augmenter la couverture",
            action=fallback_dilate,
            priority=1
        ),
        FallbackStrategy(
            name="erode",
            description="Éroder le masque pour réduire la couverture",
            action=fallback_erode,
            priority=2
        ),
        FallbackStrategy(
            name="semantic_only",
            description="Utiliser uniquement le masque sémantique",
            action=fallback_semantic_only,
            priority=3
        ),
        FallbackStrategy(
            name="clean_fragments",
            description="Nettoyer les petits fragments",
            action=fallback_clean_fragments,
            priority=4
        ),
        FallbackStrategy(
            name="default_mask",
            description="Utiliser un masque par défaut",
            action=fallback_default_mask,
            priority=5
        )
    ]


def fallback_dilate(mask: Image.Image, **kwargs) -> Image.Image:
    """Fallback: Dilater le masque"""
    
    from .mask_refinement import dilate_mask
    
    print("      🔄 Fallback: Dilatation du masque")
    return dilate_mask(mask, iterations=5)


def fallback_erode(mask: Image.Image, **kwargs) -> Image.Image:
    """Fallback: Éroder le masque"""
    
    from .mask_refinement import erode_mask
    
    print("      🔄 Fallback: Érosion du masque")
    return erode_mask(mask, iterations=3)


def fallback_semantic_only(mask: Image.Image, **kwargs) -> Image.Image:
    """Fallback: Utiliser le masque sémantique"""
    
    semantic_mask = kwargs.get("semantic_mask")
    
    if semantic_mask is not None:
        print("      🔄 Fallback: Utilisation masque sémantique")
        return semantic_mask
    
    return mask


def fallback_clean_fragments(mask: Image.Image, **kwargs) -> Image.Image:
    """Fallback: Nettoyer les petits fragments"""
    
    from .mask_refinement import clean_mask_morphology
    
    print("      🔄 Fallback: Nettoyage des fragments")
    return clean_mask_morphology(mask, min_area=500)


def fallback_default_mask(mask: Image.Image, **kwargs) -> Image.Image:
    """Fallback: Créer un masque par défaut (centre de l'image)"""
    
    print("      🔄 Fallback: Masque par défaut (centre)")
    
    width, height = mask.size
    
    # Créer un masque elliptique au centre
    default = Image.new("L", (width, height), 0)
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(default)
    
    # Ellipse couvrant le centre
    margin_x = width // 4
    margin_y = height // 4
    
    draw.ellipse(
        [margin_x, margin_y, width - margin_x, height - margin_y],
        fill=255
    )
    
    return default


# =====================================================
# AUTO-CORRECTION
# =====================================================

def auto_correct_mask(
    mask: Image.Image,
    validation_result: ValidationResult,
    semantic_mask: Optional[Image.Image] = None,
    max_attempts: int = 3
) -> ValidationResult:
    """
    Tente de corriger automatiquement un masque invalide
    
    Args:
        mask: Masque à corriger
        validation_result: Résultat de validation initial
        semantic_mask: Masque sémantique de fallback
        max_attempts: Nombre max de tentatives
    
    Returns:
        Nouveau ValidationResult (possiblement corrigé)
    """
    
    if validation_result.is_valid:
        return validation_result
    
    print(f"   🔧 Auto-correction du masque ({validation_result.status.value})")
    
    current_mask = mask
    strategies = create_default_fallback_strategies()
    
    # Choisir les stratégies selon le problème
    if validation_result.status == ValidationStatus.TOO_SMALL:
        applicable = [s for s in strategies if s.name in ["dilate", "semantic_only"]]
    elif validation_result.status == ValidationStatus.TOO_LARGE:
        applicable = [s for s in strategies if s.name in ["erode", "clean_fragments"]]
    elif validation_result.status == ValidationStatus.EMPTY:
        applicable = [s for s in strategies if s.name in ["semantic_only", "default_mask"]]
    elif validation_result.status == ValidationStatus.FRAGMENTED:
        applicable = [s for s in strategies if s.name in ["clean_fragments", "erode"]]
    else:
        applicable = strategies
    
    # Trier par priorité
    applicable.sort(key=lambda s: s.priority)
    
    # Tenter chaque stratégie
    for attempt, strategy in enumerate(applicable[:max_attempts]):
        
        print(f"      Tentative {attempt + 1}: {strategy.name}")
        
        try:
            corrected = strategy.action(
                current_mask,
                semantic_mask=semantic_mask
            )
            
            # Re-valider
            new_result = validate_mask(corrected)
            
            if new_result.is_valid:
                print(f"   ✅ Correction réussie avec {strategy.name}")
                return new_result
            
            # Continuer avec le masque corrigé
            current_mask = corrected
            
        except Exception as e:
            print(f"      ⚠️ Erreur: {e}")
            continue
    
    # Échec de toutes les stratégies
    print(f"   ⚠️ Impossible de corriger automatiquement")
    
    return ValidationResult(
        status=ValidationStatus.INVALID,
        coverage=validation_result.coverage,
        message="Correction automatique échouée",
        mask=current_mask,
        is_valid=False,
        suggestions=[
            "Vérifier manuellement le masque",
            "Ajuster les paramètres de segmentation",
            "Changer la cible de segmentation"
        ]
    )


# =====================================================
# MÉTRIQUES AVANCÉES
# =====================================================

def compute_mask_metrics(mask: Image.Image) -> dict:
    """
    Calcule des métriques détaillées sur le masque
    
    Returns:
        Dict avec coverage, fragments, compactness, etc.
    """
    
    from scipy import ndimage
    
    mask_array = np.array(mask) > 127
    
    # Couverture
    coverage = np.sum(mask_array) / mask_array.size
    
    # Nombre de fragments
    labeled, num_fragments = ndimage.label(mask_array)
    
    # Plus grande région
    if num_fragments > 0:
        region_sizes = ndimage.sum(mask_array, labeled, range(1, num_fragments + 1))
        largest_region_ratio = max(region_sizes) / mask_array.size
    else:
        largest_region_ratio = 0
    
    # Compacité (périmètre² / aire)
    # Plus la valeur est basse, plus la forme est compacte
    perimeter = compute_perimeter(mask_array)
    area = np.sum(mask_array)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
    
    # Centre de masse
    if area > 0:
        center_y, center_x = ndimage.center_of_mass(mask_array)
    else:
        center_x, center_y = mask.size[0] // 2, mask.size[1] // 2
    
    return {
        "coverage": coverage,
        "num_fragments": num_fragments,
        "largest_region_ratio": largest_region_ratio,
        "compactness": compactness,
        "center_x": center_x,
        "center_y": center_y,
        "width": mask.size[0],
        "height": mask.size[1]
    }


def compute_perimeter(mask_array: np.ndarray) -> float:
    """Calcule le périmètre approximatif du masque"""
    
    from scipy import ndimage
    
    # Gradient pour détecter les bords
    grad_x = ndimage.sobel(mask_array.astype(float), axis=1)
    grad_y = ndimage.sobel(mask_array.astype(float), axis=0)
    
    # Magnitude du gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return np.sum(magnitude > 0)


# =====================================================
# VALIDATION PAR RÈGLES MÉTIER
# =====================================================

def validate_for_inpainting(mask: Image.Image) -> ValidationResult:
    """Validation spécifique pour l'inpainting"""
    
    result = validate_mask(
        mask,
        min_coverage=0.02,   # Au moins 2%
        max_coverage=0.70,   # Pas plus de 70%
        check_fragmentation=False  # Inpainting tolère les fragments
    )
    
    if result.is_valid:
        result.message = "Masque valide pour inpainting"
    
    return result


def validate_for_generation(mask: Image.Image) -> ValidationResult:
    """Validation spécifique pour la génération complète"""
    
    result = validate_mask(
        mask,
        min_coverage=0.1,    # Au moins 10%
        max_coverage=0.50,   # Pas plus de 50%
        check_fragmentation=True
    )
    
    if result.is_valid:
        result.message = "Masque valide pour génération"
    
    return result


def validate_for_style_transfer(mask: Image.Image) -> ValidationResult:
    """Validation spécifique pour le style transfer"""
    
    result = validate_mask(
        mask,
        min_coverage=0.05,   # Au moins 5%
        max_coverage=0.80,   # Peut aller jusqu'à 80%
        check_fragmentation=False
    )
    
    if result.is_valid:
        result.message = "Masque valide pour style transfer"
    
    return result
