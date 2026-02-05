# Grounding DINO - Détection d'objets par text prompt
# Pour détecter les ouvertures (fenêtres, portes) non détectées par OneFormer

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import warnings

# Cache global
_grounding_dino_model = None
_grounding_dino_processor = None


def load_grounding_dino():
    """
    Charge Grounding DINO pour la détection d'objets par text prompt
    """
    global _grounding_dino_model, _grounding_dino_processor
    
    if _grounding_dino_model is not None:
        print("   ♻️  Grounding DINO déjà chargé (cache)")
        return _grounding_dino_model, _grounding_dino_processor
    
    print("   🎯 Chargement de Grounding DINO...")
    
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        
        model_id = "IDEA-Research/grounding-dino-base"
        
        _grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
        _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to("cuda").to(torch.float32)  # Force float32 pour éviter les erreurs de mixed precision
        
        print(f"   ✅ Grounding DINO chargé ({model_id})")
        
    except Exception as e:
        print(f"   ⚠️  Erreur lors du chargement de Grounding DINO: {e}")
        print(f"   💡 Installez avec: pip install transformers")
        return None, None
    
    return _grounding_dino_model, _grounding_dino_processor


def detect_with_grounding_dino(
    image: Image.Image,
    text_prompt: str,
    confidence_threshold: float = 0.25,
    box_threshold: float = 0.25,
    model = None,
    processor = None
) -> Tuple[List[np.ndarray], List[float], List[str]]:
    """
    Détecte des objets dans une image avec un text prompt
    
    Args:
        image: Image PIL
        text_prompt: Prompt texte (ex: "window . door . glass window")
        confidence_threshold: Seuil de confiance minimum
        box_threshold: Seuil pour les bounding boxes
        model: Modèle Grounding DINO (chargé si None)
        processor: Processeur (chargé si None)
    
    Returns:
        boxes: Liste de bounding boxes [x1, y1, x2, y2]
        scores: Liste de scores de confiance
        labels: Liste de labels détectés
    
    Example:
        >>> boxes, scores, labels = detect_with_grounding_dino(
        ...     image, 
        ...     "window . door . glass window"
        ... )
    """
    
    # Charger le modèle si nécessaire
    if model is None or processor is None:
        model, processor = load_grounding_dino()
        
        if model is None:
            return [], [], []
    
    # Préparer les inputs
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    ).to("cuda")
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement
    try:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
    except TypeError:
        # Nouvelle API sans box_threshold
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=confidence_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
        
        # Filtrer par score
        mask = results["scores"] >= box_threshold
        results = {
            "boxes": results["boxes"][mask],
            "scores": results["scores"][mask],
            "labels": [label for i, label in enumerate(results["labels"]) if mask[i]]
        }
    
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    
    return boxes, scores, labels


def boxes_to_mask(
    boxes: List[np.ndarray],
    image_size: Tuple[int, int],
    dilation: int = 10
) -> Image.Image:
    """
    Convertit des bounding boxes en masque binaire
    
    Args:
        boxes: Liste de boxes [x1, y1, x2, y2]
        image_size: (width, height)
        dilation: Pixels à ajouter autour des boxes
    
    Returns:
        Masque PIL (blanc pour les objets détectés)
    """
    
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Dilater la box
        x1 = max(0, int(x1) - dilation)
        y1 = max(0, int(y1) - dilation)
        x2 = min(width, int(x2) + dilation)
        y2 = min(height, int(y2) + dilation)
        
        # Remplir la région
        mask[y1:y2, x1:x2] = 255
    
    return Image.fromarray(mask, mode="L")


def detect_openings(
    image: Image.Image,
    detect_windows: bool = True,
    detect_doors: bool = True,
    confidence_threshold: float = 0.25,
    model = None,
    processor = None
) -> Tuple[Image.Image, dict]:
    """
    Détecte les ouvertures (fenêtres et portes) dans une image
    
    Args:
        image: Image PIL
        detect_windows: Détecter les fenêtres
        detect_doors: Détecter les portes
        confidence_threshold: Seuil de confiance
        model: Modèle Grounding DINO (optionnel)
        processor: Processeur (optionnel)
    
    Returns:
        mask: Masque combiné des ouvertures
        metadata: Dictionnaire avec les détails de détection
    
    Example:
        >>> mask, metadata = detect_openings(image)
        >>> print(f"Fenêtres: {metadata['num_windows']}, Portes: {metadata['num_doors']}")
    """
    
    print("   🎯 Détection des ouvertures avec Grounding DINO...")
    
    # Charger le modèle
    if model is None or processor is None:
        model, processor = load_grounding_dino()
        
        if model is None:
            print("      ⚠️  Grounding DINO non disponible")
            return Image.new("L", image.size, 0), {}
    
    all_boxes = []
    metadata = {
        "num_windows": 0,
        "num_doors": 0,
        "window_boxes": [],
        "door_boxes": [],
        "window_scores": [],
        "door_scores": []
    }
    
    # Détecter les fenêtres
    if detect_windows:
        window_prompt = "window . glass window . windowpane . french window"
        boxes, scores, labels = detect_with_grounding_dino(
            image, 
            window_prompt,
            confidence_threshold=confidence_threshold,
            model=model,
            processor=processor
        )
        
        if len(boxes) > 0:
            all_boxes.extend(boxes)
            metadata["num_windows"] = len(boxes)
            metadata["window_boxes"] = boxes.tolist()
            metadata["window_scores"] = scores.tolist()
            print(f"      ✅ {len(boxes)} fenêtre(s) détectée(s)")
    
    # Détecter les portes
    if detect_doors:
        door_prompt = "door . entrance door . doorway . french door . sliding door"
        boxes, scores, labels = detect_with_grounding_dino(
            image,
            door_prompt,
            confidence_threshold=confidence_threshold,
            model=model,
            processor=processor
        )
        
        if len(boxes) > 0:
            all_boxes.extend(boxes)
            metadata["num_doors"] = len(boxes)
            metadata["door_boxes"] = boxes.tolist()
            metadata["door_scores"] = scores.tolist()
            print(f"      ✅ {len(boxes)} porte(s) détectée(s)")
    
    # Convertir en masque
    if all_boxes:
        mask = boxes_to_mask(all_boxes, image.size, dilation=10)
        coverage = np.sum(np.array(mask) > 127) / (image.size[0] * image.size[1])
        metadata["coverage"] = coverage
        print(f"      📊 Couverture totale: {coverage*100:.2f}%")
    else:
        mask = Image.new("L", image.size, 0)
        metadata["coverage"] = 0.0
        print(f"      ⚠️  Aucune ouverture détectée")
    
    return mask, metadata


def refine_with_sam2(
    image: Image.Image,
    boxes: List[np.ndarray],
    sam2_model = None,
    sam2_processor = None
) -> Image.Image:
    """
    Raffine les détections de Grounding DINO avec SAM2
    
    Args:
        image: Image PIL
        boxes: Bounding boxes de Grounding DINO
        sam2_model: Modèle SAM2 (chargé si None)
        sam2_processor: Processeur SAM2 (chargé si None)
    
    Returns:
        Masque raffiné par SAM2
    """
    
    if not boxes or len(boxes) == 0:
        return Image.new("L", image.size, 0)
    
    print(f"   ✨ Raffinement SAM2 de {len(boxes)} détections...")
    
    # Charger SAM2 si nécessaire
    if sam2_model is None or sam2_processor is None:
        from models.sam2 import load_sam2
        sam2_model, sam2_processor = load_sam2()
    
    # Convertir boxes en format SAM2
    # SAM2 attend [[[[x1, y1, x2, y2]]]]
    boxes_list = [[[[int(b[0]), int(b[1]), int(b[2]), int(b[3])]] for b in boxes]]
    
    # Préparer inputs
    inputs = sam2_processor(
        image,
        input_boxes=boxes_list,
        return_tensors="pt"
    )
    
    # Déplacer sur GPU
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to("cuda")
    
    # Inférence
    with torch.no_grad():
        outputs = sam2_model(**inputs)
    
    # Extraire les masques
    masks = outputs.pred_masks.squeeze().cpu().numpy()
    
    # Combiner tous les masques
    if masks.ndim == 3:
        combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8) * 255
    else:
        combined_mask = (masks > 0.5).astype(np.uint8) * 255
    
    refined_mask = Image.fromarray(combined_mask, mode="L")
    
    # Redimensionner si nécessaire
    if refined_mask.size != image.size:
        refined_mask = refined_mask.resize(image.size, Image.BILINEAR)
    
    coverage = np.sum(np.array(refined_mask) > 127) / (image.size[0] * image.size[1])
    print(f"   ✅ Raffinement terminé ({coverage*100:.2f}% couverture)")
    
    return refined_mask
