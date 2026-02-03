# SAM2 - Segment Anything Model 2 (Meta 2024)
import torch
import numpy as np
from PIL import Image

# Global model cache
_sam2_model = None
_sam2_processor = None


def load_sam2():
    """
    Charge SAM2 (Segment Anything Model 2) de Meta
    Le vrai SAM2 sorti en 2024
    """
    global _sam2_model, _sam2_processor
    
    if _sam2_model is not None:
        print("   ♻️  SAM2 déjà chargé (cache)")
        return _sam2_model, _sam2_processor
    
    print("   🧠 Chargement de SAM2 (Meta)...")
    
    from transformers import Sam2Model, Sam2Processor
    
    # SAM2 - Le vrai modèle Meta 2024
    _sam2_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    _sam2_model = Sam2Model.from_pretrained(
        "facebook/sam2-hiera-large",
        torch_dtype=torch.float16
    ).to("cuda")
    
    print("   ✅ SAM2 chargé (facebook/sam2-hiera-large)")
    
    return _sam2_model, _sam2_processor


def load_segformer_floor():
    """
    Charge SegFormer pour la détection automatique du sol
    Plus adapté pour la segmentation sémantique des intérieurs
    """
    print("   🏠 Chargement de SegFormer (ADE20K)...")
    
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        torch_dtype=torch.float32  # float32 pour éviter les erreurs de type
    ).to("cuda")
    
    print("   ✅ SegFormer chargé")
    
    return model, processor


def segment_floor_auto(image: Image.Image, save_path: str = None) -> Image.Image:
    """
    Segmente automatiquement le sol avec SegFormer
    Retourne un masque binaire du sol
    
    Args:
        image: Image PIL d'entrée
        save_path: Chemin pour sauvegarder le masque (optionnel)
    
    Returns:
        Masque PIL (blanc = sol, noir = reste)
    """
    import torch.nn.functional as F
    
    print("   🎯 Détection automatique du sol...")
    
    model, processor = load_segformer_floor()
    
    # Préparation - s'assurer que les inputs sont en float32
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.float32) for k, v in inputs.items()}
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Redimensionner à la taille originale
    upsampled = F.interpolate(
        logits,
        size=image.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False
    )
    
    # Obtenir les classes prédites
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    
    # Classes ADE20K pour le sol: 
    # 3 = floor, 4 = ceiling, 28 = rug, 29 = carpet
    floor_classes = [3, 28, 29]  # floor, rug, carpet
    
    # Créer le masque binaire
    mask = np.isin(pred, floor_classes).astype(np.uint8) * 255
    
    mask_image = Image.fromarray(mask, mode="L")
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque sol sauvegardé: {save_path}")
    
    # Nettoyer la mémoire
    del model, processor
    torch.cuda.empty_cache()
    
    print(f"   ✅ Sol segmenté ({np.sum(mask > 0) / mask.size * 100:.1f}% de l'image)")
    
    return mask_image


def segment_with_points_sam2(
    image: Image.Image, 
    points: list[tuple[int, int]],
    labels: list[int] = None,
    save_path: str = None
) -> Image.Image:
    """
    Segmente avec SAM2 en utilisant des points de référence
    
    Args:
        image: Image PIL d'entrée
        points: Liste de points (x, y) pour guider la segmentation
        labels: Liste de labels (1 = inclure, 0 = exclure)
        save_path: Chemin pour sauvegarder le masque
    
    Returns:
        Masque PIL de la région segmentée
    """
    print(f"   🎯 Segmentation SAM2 avec {len(points)} points...")
    
    model, processor = load_sam2()
    
    if labels is None:
        labels = [1] * len(points)  # Tous les points sont inclusifs
    
    # Préparer les entrées pour SAM2
    inputs = processor(
        images=image,
        input_points=[points],
        input_labels=[labels],
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    # Inférence SAM2
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement SAM2
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )
    
    # Prendre le meilleur masque (score le plus élevé)
    scores = outputs.iou_scores.cpu().numpy()[0]
    best_idx = scores.argmax()
    
    mask = masks[0][0][best_idx].cpu().numpy().astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque SAM2 sauvegardé: {save_path}")
    
    print(f"   ✅ Segmentation SAM2 terminée (score IoU: {scores[best_idx]:.3f})")
    
    return mask_image


def segment_with_box_sam2(
    image: Image.Image,
    box: tuple[int, int, int, int],
    save_path: str = None
) -> Image.Image:
    """
    Segmente avec SAM2 en utilisant une bounding box
    
    Args:
        image: Image PIL d'entrée
        box: Bounding box (x1, y1, x2, y2)
        save_path: Chemin pour sauvegarder le masque
    
    Returns:
        Masque PIL de la région segmentée
    """
    print(f"   📦 Segmentation SAM2 avec box {box}...")
    
    model, processor = load_sam2()
    
    # Préparer les entrées avec bounding box pour SAM2
    inputs = processor(
        images=image,
        input_boxes=[[[list(box)]]],
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    # Inférence SAM2
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement SAM2
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )
    
    # Prendre le meilleur masque
    scores = outputs.iou_scores.cpu().numpy()[0]
    best_idx = scores.argmax()
    
    mask = masks[0][0][best_idx].cpu().numpy().astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque SAM2 sauvegardé: {save_path}")
    
    print(f"   ✅ Segmentation terminée (score IoU: {scores[best_idx]:.3f})")
    
    return mask_image


def dilate_mask(mask: Image.Image, iterations: int = 5) -> Image.Image:
    """
    Dilate le masque pour couvrir les bords
    """
    import cv2
    
    mask_np = np.array(mask)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=iterations)
    
    return Image.fromarray(dilated, mode="L")


def feather_mask(mask: Image.Image, radius: int = 10) -> Image.Image:
    """
    Applique un feathering (adoucissement des bords) au masque
    """
    import cv2
    
    mask_np = np.array(mask)
    blurred = cv2.GaussianBlur(mask_np, (radius * 2 + 1, radius * 2 + 1), 0)
    
    return Image.fromarray(blurred, mode="L")
