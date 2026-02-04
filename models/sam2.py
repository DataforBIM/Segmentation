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
    # Format requis: [batch, object, point, coordinates] = 4 niveaux
    # Pour une image avec un objet: [[[[x1, y1], [x2, y2], ...]]]
    # Convertir points en liste de listes [x, y]
    points_formatted = [[list(p) for p in points]]  # [[x1,y1], [x2,y2], ...]
    labels_formatted = [labels]  # [1, 1, ...]
    
    inputs = processor(
        images=image,
        input_points=[points_formatted],  # [batch [object [points]]]
        input_labels=[labels_formatted],  # [batch [labels]]
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    # Inférence SAM2
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement SAM2 - nouvelle API
    # pred_masks shape: [batch, num_objects, num_masks, H, W]
    masks = outputs.pred_masks  # Garder toutes les dimensions
    
    # Prendre le meilleur masque basé sur iou_scores
    # iou_scores shape: [batch, num_objects, num_masks]
    scores = outputs.iou_scores.cpu().numpy()
    
    # Aplatir pour trouver le meilleur
    scores_flat = scores.flatten()
    best_idx = scores_flat.argmax()
    best_score = scores_flat[best_idx]
    
    # Extraire le masque correspondant
    # On suppose batch=1 et on prend le meilleur masque du premier objet
    mask_tensor = masks[0, 0, best_idx % masks.shape[2]]  # [H, W]
    mask = (mask_tensor > 0.5).float().cpu().numpy().astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    
    # Redimensionner si nécessaire
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.NEAREST)
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque SAM2 sauvegardé: {save_path}")
    
    print(f"   ✅ Segmentation SAM2 terminée (score IoU: {best_score:.3f})")
    
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


# ============================================================
# GROUNDED SAM2 - Détection automatique + Segmentation
# ============================================================

_grounding_dino_model = None
_grounding_dino_processor = None


def load_grounding_dino():
    """
    Charge Grounding DINO pour la détection basée sur le texte
    """
    global _grounding_dino_model, _grounding_dino_processor
    
    if _grounding_dino_model is not None:
        print("   ♻️  Grounding DINO déjà chargé (cache)")
        return _grounding_dino_model, _grounding_dino_processor
    
    print("   🔍 Chargement de Grounding DINO...")
    
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    model_id = "IDEA-Research/grounding-dino-base"
    
    _grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
    _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        torch_dtype=torch.float32  # float32 pour éviter les erreurs de dtype
    ).to("cuda")
    
    print("   ✅ Grounding DINO chargé")
    
    return _grounding_dino_model, _grounding_dino_processor


def detect_objects_grounding_dino(
    image: Image.Image,
    text_prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25
) -> list[dict]:
    """
    Détecte des objets dans l'image à partir d'une description textuelle
    
    Args:
        image: Image PIL
        text_prompt: Description de ce qu'on cherche (ex: "cat ears")
        box_threshold: Seuil de confiance pour les boxes
        text_threshold: Seuil de confiance pour le texte
    
    Returns:
        Liste de détections avec boxes et scores
    """
    model, processor = load_grounding_dino()
    
    # Préparer les entrées - s'assurer qu'elles sont en float32
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    )
    # Convertir tous les tenseurs en float32 et les mettre sur CUDA
    inputs = {k: v.to("cuda", dtype=torch.float32) if v.dtype in [torch.float16, torch.float32] else v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-traitement - nouvelle API
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        target_sizes=[image.size[::-1]]  # (height, width)
    )[0]
    
    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        # Filtrer par score
        if float(score.cpu()) >= box_threshold:
            box_coords = box.cpu().numpy().tolist()
            detections.append({
                "box": box_coords,  # [x1, y1, x2, y2]
                "score": float(score.cpu()),
                "label": label
            })
    
    return detections


def segment_animal_part(
    image: Image.Image,
    target: str,
    save_path: str = None
) -> Image.Image:
    """
    Segmente automatiquement une partie d'animal avec Grounded SAM2
    
    Args:
        image: Image PIL
        target: Partie à segmenter ("ears", "eyes", "fur", "tail", "paws", "nose", "body", "background")
        save_path: Chemin pour sauvegarder le masque
    
    Returns:
        Masque PIL de la région segmentée
    """
    # CAS SPÉCIAL: Si target = "background", on segmente l'animal puis on inverse
    if target == "background":
        print(f"   🔄 Mode 'background': segmentation de l'animal puis inversion du masque")
        # Segmenter l'animal entier (body)
        animal_mask = segment_animal_part(image, target="body", save_path=None)
        # Inverser le masque (blanc <-> noir)
        inverted_mask = Image.eval(animal_mask, lambda x: 255 - x)
        if save_path:
            inverted_mask.save(save_path)
        return inverted_mask
    
    # Mapping des cibles vers les prompts de détection
    target_prompts = {
        "ears": "cat ear. dog ear. animal ear.",
        "eyes": "cat eye. dog eye. animal eye.",
        "fur": "cat fur. dog fur. animal body. cat body.",
        "tail": "cat tail. dog tail. animal tail.",
        "paws": "cat paw. dog paw. animal paw. cat leg.",
        "nose": "cat nose. dog nose. animal nose. cat muzzle.",
        "body": "cat. dog. animal. pet."
    }
    
    # Limites de taille des boxes (% de l'image) selon la partie
    max_box_size = {
        "ears": 0.15,   # Les oreilles < 15% de l'image
        "eyes": 0.08,   # Les yeux < 8% de l'image
        "nose": 0.05,   # Le nez < 5% de l'image
        "paws": 0.20,   # Les pattes < 20%
        "tail": 0.25,   # La queue < 25%
        "fur": 0.90,    # Le pelage peut être grand
        "body": 0.95    # Le corps entier peut être très grand
    }
    
    # Nombre max de détections selon la partie
    max_detections = {
        "ears": 2,      # 2 oreilles
        "eyes": 2,      # 2 yeux
        "nose": 1,      # 1 nez
        "paws": 4,      # 4 pattes
        "tail": 1,      # 1 queue
        "fur": 1,       # 1 corps
        "body": 1       # 1 animal entier
    }
    
    text_prompt = target_prompts.get(target, f"{target}.")
    max_size = max_box_size.get(target, 0.30)
    max_det = max_detections.get(target, 4)
    
    print(f"   🔍 Détection automatique: '{text_prompt}'")
    
    # Étape 1: Détection avec Grounding DINO
    detections = detect_objects_grounding_dino(
        image,
        text_prompt,
        box_threshold=0.25,  # Seuil plus strict
        text_threshold=0.25
    )
    
    if not detections:
        print(f"   ⚠️  Aucune détection pour '{target}', utilisation de points par défaut")
        return _segment_with_default_points(image, target, save_path)
    
    # Filtrer les détections par taille de box
    w, h = image.size
    image_area = w * h
    
    filtered_detections = []
    for det in detections:
        box = det["box"]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        box_area = box_w * box_h
        box_ratio = box_area / image_area
        
        # Exclure les boxes trop grandes (probablement tout l'animal)
        if box_ratio <= max_size:
            det["box_ratio"] = box_ratio
            det["center_x"] = (box[0] + box[2]) / 2
            det["center_y"] = (box[1] + box[3]) / 2
            filtered_detections.append(det)
        else:
            print(f"   ⚠️  Box ignorée (trop grande: {box_ratio*100:.1f}% > {max_size*100:.0f}%)")
    
    # Trier par score
    filtered_detections.sort(key=lambda x: x["score"], reverse=True)
    
    # NMS simple: garder des boxes de zones différentes
    # Pour les yeux: une à gauche, une à droite
    final_detections = []
    min_distance = w * 0.1  # Distance min entre centres (10% de la largeur)
    
    for det in filtered_detections:
        is_duplicate = False
        for kept in final_detections:
            # Calculer la distance entre centres
            dist = ((det["center_x"] - kept["center_x"])**2 + 
                    (det["center_y"] - kept["center_y"])**2) ** 0.5
            if dist < min_distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_detections.append(det)
            if len(final_detections) >= max_det:
                break
    
    if not final_detections:
        print(f"   ⚠️  Toutes les détections filtrées, utilisation de points par défaut")
        return _segment_with_default_points(image, target, save_path)
    
    print(f"   ✅ {len(final_detections)} détection(s) retenue(s) (sur {len(detections)} trouvées)")
    for i, det in enumerate(final_detections):
        print(f"      {i+1}. {det['label']} (score: {det['score']:.2f}, pos: x={det['center_x']:.0f})")
    
    # Étape 2: Segmentation avec SAM2 en utilisant les boxes filtrées
    model, processor = load_sam2()
    
    boxes = [det["box"] for det in final_detections]
    
    # Préparer les entrées SAM2 avec les boxes
    inputs = processor(
        images=image,
        input_boxes=[boxes],
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtenir les tailles originales
    original_sizes = inputs["original_sizes"].tolist()
    
    # Post-process les masques - API actuelle: (masks, original_sizes, ...)
    masks = processor.post_process_masks(
        outputs.pred_masks,
        original_sizes,
        binarize=False
    )
    
    # Fusionner les masques de toutes les détections
    combined_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    
    for i in range(len(boxes)):
        scores = outputs.iou_scores[0][i].cpu().numpy()
        best_idx = scores.argmax()
        mask = masks[0][i][best_idx].cpu().numpy().astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, mask)
    
    mask_image = Image.fromarray(combined_mask, mode="L")
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque Grounded-SAM2 sauvegardé: {save_path}")
    
    coverage = np.sum(combined_mask > 0) / combined_mask.size * 100
    print(f"   ✅ Segmentation automatique terminée ({coverage:.1f}% de l'image)")
    
    return mask_image


def _segment_with_default_points(
    image: Image.Image,
    target: str,
    save_path: str = None
) -> Image.Image:
    """
    Fallback: segmente avec des points par défaut si Grounding DINO échoue
    """
    w, h = image.size
    
    # Points par défaut selon la partie
    default_points = {
        "ears": [(int(w * 0.3), int(h * 0.12)), (int(w * 0.7), int(h * 0.12))],
        "eyes": [(int(w * 0.35), int(h * 0.35)), (int(w * 0.65), int(h * 0.35))],
        "fur": [(int(w * 0.5), int(h * 0.5))],
        "tail": [(int(w * 0.15), int(h * 0.5))],
        "paws": [(int(w * 0.3), int(h * 0.85)), (int(w * 0.7), int(h * 0.85))],
        "nose": [(int(w * 0.5), int(h * 0.45))]
    }
    
    points = default_points.get(target, [(w // 2, h // 2)])
    
    print(f"   🎯 Utilisation de {len(points)} points par défaut pour '{target}'")
    
    return segment_with_points_sam2(image, points, save_path=save_path)


# ============================================================
# SEGMENTATION PAR TYPE DE SCÈNE
# ============================================================

def segment_interior_element(
    image: Image.Image,
    target: str,
    save_path: str = None
) -> Image.Image:
    """
    Segmente un élément d'intérieur (floor, wall, ceiling, furniture)
    
    Args:
        image: Image PIL
        target: Élément à segmenter
        save_path: Chemin de sauvegarde
    
    Returns:
        Masque PIL
    """
    # Prompts de détection pour Grounding DINO
    interior_prompts = {
        "floor": "floor. ground. tile. wood floor. carpet.",
        "wall": "wall. interior wall. painted wall.",
        "ceiling": "ceiling. roof interior.",
        "furniture": "furniture. sofa. table. chair. bed. desk.",
        "window": "window. glass window. window frame.",
        "door": "door. wooden door. entrance.",
    }
    
    # Taille max des boxes (% de l'image)
    max_sizes = {
        "floor": 0.60,      # Le sol peut couvrir 60%
        "wall": 0.70,       # Les murs peuvent couvrir 70%
        "ceiling": 0.40,    # Le plafond ~40%
        "furniture": 0.50,  # Meubles ~50%
        "window": 0.30,     # Fenêtres ~30%
        "door": 0.25,       # Portes ~25%
    }
    
    text_prompt = interior_prompts.get(target, f"{target}.")
    max_size = max_sizes.get(target, 0.50)
    
    print(f"   🏠 Segmentation intérieur: '{target}'")
    
    return _segment_with_grounded_sam2(image, text_prompt, max_size, save_path)


def segment_exterior_element(
    image: Image.Image,
    target: str,
    save_path: str = None
) -> Image.Image:
    """
    Segmente un élément d'extérieur (sky, ground, vegetation, building)
    """
    exterior_prompts = {
        "sky": "sky. blue sky. clouds.",
        "ground": "ground. grass. pavement. road.",
        "vegetation": "tree. trees. plant. plants. grass. vegetation. forest.",
        "building": "building. house. facade. architecture.",
        "road": "road. street. asphalt. pavement.",
    }
    
    max_sizes = {
        "sky": 0.70,        # Le ciel peut couvrir 70%
        "ground": 0.50,     # Le sol ~50%
        "vegetation": 0.60, # Végétation ~60%
        "building": 0.60,   # Bâtiments ~60%
        "road": 0.40,       # Route ~40%
    }
    
    text_prompt = exterior_prompts.get(target, f"{target}.")
    max_size = max_sizes.get(target, 0.50)
    
    print(f"   🌳 Segmentation extérieur: '{target}'")
    
    return _segment_with_grounded_sam2(image, text_prompt, max_size, save_path)


def segment_portrait_element(
    image: Image.Image,
    target: str,
    save_path: str = None
) -> Image.Image:
    """
    Segmente un élément de portrait (face, hair, eyes, lips, clothing)
    """
    portrait_prompts = {
        "face": "face. human face. skin.",
        "hair": "hair. hairstyle. head hair.",
        "eyes": "human eye. eyes.",
        "lips": "lips. mouth.",
        "skin": "skin. human skin.",
        "clothing": "clothing. shirt. dress. jacket. clothes.",
    }
    
    max_sizes = {
        "face": 0.40,       # Visage ~40%
        "hair": 0.25,       # Cheveux ~25%
        "eyes": 0.05,       # Yeux ~5%
        "lips": 0.03,       # Lèvres ~3%
        "skin": 0.50,       # Peau ~50%
        "clothing": 0.60,   # Vêtements ~60%
    }
    
    max_detections = {
        "face": 1,
        "hair": 1,
        "eyes": 2,
        "lips": 1,
        "skin": 1,
        "clothing": 3,
    }
    
    text_prompt = portrait_prompts.get(target, f"{target}.")
    max_size = max_sizes.get(target, 0.30)
    max_det = max_detections.get(target, 2)
    
    print(f"   👤 Segmentation portrait: '{target}'")
    
    return _segment_with_grounded_sam2(image, text_prompt, max_size, save_path, max_det)


def segment_aerial_elements(
    image: Image.Image,
    save_path: str = None
) -> dict:
    """
    Segmente TOUS les éléments architecturaux d'une vue aérienne séparément
    Pour que SDXL puisse traiter chaque élément individuellement
    
    Returns:
        Dict avec:
        - "masks": Dict de masques par élément {"building": mask, "roof": mask, ...}
        - "combined_mask": Masque combiné de tous les éléments
        - "elements_found": Liste des éléments détectés
    """
    print(f"   🚁 Segmentation aérienne multi-éléments avec SAM2")
    
    # Tous les éléments à détecter dans une vue aérienne
    aerial_elements = {
        # "building": "building. house. construction. structure.",  # DÉSACTIVÉ
        "walls": "wall. exterior wall. facade. building wall. side wall.",
        "ornementation": "ornementation. decoration. architectural detail. ornament. molding. trim.",
        "roof": "roof. rooftop. building roof.",
        "door": "door. entrance. building entrance.",
        "road": "road. street. pavement. asphalt. roadway. highway. avenue. boulevard. driveway.",
        "road_markings": "road marking. road line. street marking. lane marking. crosswalk. zebra crossing. painted line.",
        "sidewalk": "sidewalk. footpath. pavement. walkway. pedestrian path. pathway. footway.",
        "car": "car. vehicle. automobile.",
        "vegetation": "tree. vegetation. plant. grass.",
        "parking": "parking lot. parking space. parking area. car park.",
    }
    
    # Configuration par élément
    element_config = {
        # "building": {"max_size": 0.80, "max_det": 20, "threshold": 0.30},  # DÉSACTIVÉ
        "walls": {"max_size": 0.70, "max_det": 25, "threshold": 0.28},
        "ornementation": {"max_size": 0.15, "max_det": 40, "threshold": 0.25},
        "roof": {"max_size": 0.60, "max_det": 20, "threshold": 0.30},
        "door": {"max_size": 0.10, "max_det": 30, "threshold": 0.25},
        "road": {"max_size": 0.85, "max_det": 15, "threshold": 0.18},  # Seuil réduit + max_size augmenté
        "road_markings": {"max_size": 0.05, "max_det": 50, "threshold": 0.20},
        "sidewalk": {"max_size": 0.50, "max_det": 15, "threshold": 0.18},  # Seuil réduit + max_size augmenté
        "car": {"max_size": 0.10, "max_det": 30, "threshold": 0.35},
        "vegetation": {"max_size": 0.50, "max_det": 15, "threshold": 0.30},
        "parking": {"max_size": 0.50, "max_det": 15, "threshold": 0.20},  # Seuil réduit + max_size augmenté
    }
    
    w, h = image.size
    image_area = w * h
    
    element_masks = {}
    elements_found = []
    all_detections = {}
    
    # Détecter et segmenter chaque type d'élément
    for element_name, text_prompt in aerial_elements.items():
        print(f"\n   🔍 Détection: {element_name}")
        
        config = element_config.get(element_name, {"max_size": 0.50, "max_det": 10, "threshold": 0.30})
        
        # Étape 1: Détection avec Grounding DINO
        detections = detect_objects_grounding_dino(
            image,
            text_prompt,
            box_threshold=config["threshold"],
            text_threshold=0.25
        )
        
        if not detections:
            print(f"      ⏭️  Aucune détection pour {element_name}")
            continue
        
        # Filtrer par taille
        filtered = []
        for det in detections:
            box = det["box"]
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            box_ratio = box_area / image_area
            
            if box_ratio <= config["max_size"] and box_ratio >= 0.001:  # Au moins 0.1% de l'image
                det["box_ratio"] = box_ratio
                det["element"] = element_name
                filtered.append(det)
        
        # Trier et limiter
        filtered.sort(key=lambda x: x["score"], reverse=True)
        filtered = filtered[:config["max_det"]]
        
        if not filtered:
            print(f"      ⏭️  Toutes détections filtrées pour {element_name}")
            continue
        
        all_detections[element_name] = filtered
        print(f"      ✅ {len(filtered)} {element_name}(s) détecté(s)")
        
        # Étape 2: Segmentation SAM2 pour cet élément
        model, processor = load_sam2()
        boxes = [det["box"] for det in filtered]
        
        inputs = processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt"
        ).to("cuda", torch.float16)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        original_sizes = inputs["original_sizes"].tolist()
        masks = processor.post_process_masks(
            outputs.pred_masks,
            original_sizes,
            binarize=False
        )
        
        # Combiner les masques de toutes les instances de cet élément
        element_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(len(boxes)):
            scores = outputs.iou_scores[0][i].cpu().numpy()
            best_idx = scores.argmax()
            mask = masks[0][i][best_idx].cpu().numpy().astype(np.uint8) * 255
            element_mask = np.maximum(element_mask, mask)
        
        element_mask_img = Image.fromarray(element_mask, mode="L")
        element_masks[element_name] = element_mask_img
        elements_found.append(element_name)
        
        # Sauvegarder chaque masque individuellement
        if save_path:
            element_save_path = save_path.replace(".png", f"_{element_name}.png")
            element_mask_img.save(element_save_path)
            print(f"      💾 Masque {element_name} sauvegardé")
    
    # Créer un masque combiné de tous les éléments
    if element_masks:
        combined = np.zeros((h, w), dtype=np.uint8)
        for mask_img in element_masks.values():
            mask_np = np.array(mask_img)
            combined = np.maximum(combined, mask_np)
        
        combined_mask = Image.fromarray(combined, mode="L")
        
        if save_path:
            combined_mask.save(save_path)
            print(f"\n   💾 Masque combiné sauvegardé: {save_path}")
        
        coverage = np.sum(combined > 0) / combined.size * 100
        print(f"\n   ✅ Segmentation aérienne terminée: {len(elements_found)} types d'éléments détectés")
        print(f"      Éléments: {', '.join(elements_found)}")
        print(f"      Couverture totale: {coverage:.1f}%")
        
        return {
            "masks": element_masks,
            "combined_mask": combined_mask,
            "elements_found": elements_found,
            "detections": all_detections
        }
    else:
        print(f"\n   ⚠️  Aucun élément détecté dans l'image aérienne")
        return {
            "masks": {},
            "combined_mask": None,
            "elements_found": [],
            "detections": {}
        }


def _segment_with_grounded_sam2(
    image: Image.Image,
    text_prompt: str,
    max_box_size: float = 0.50,
    save_path: str = None,
    max_detections: int = 4
) -> Image.Image:
    """
    Fonction commune pour Grounded SAM2 avec filtrage
    """
    print(f"   🔍 Détection: '{text_prompt}'")
    
    # Étape 1: Détection avec Grounding DINO
    detections = detect_objects_grounding_dino(
        image,
        text_prompt,
        box_threshold=0.25,
        text_threshold=0.25
    )
    
    if not detections:
        print(f"   ⚠️  Aucune détection, utilisation de SegFormer")
        return segment_floor_auto(image)  # Fallback
    
    # Filtrer par taille
    w, h = image.size
    image_area = w * h
    
    filtered = []
    for det in detections:
        box = det["box"]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        box_ratio = box_area / image_area
        
        if box_ratio <= max_box_size:
            det["box_ratio"] = box_ratio
            filtered.append(det)
        else:
            print(f"   ⚠️  Box ignorée (trop grande: {box_ratio*100:.1f}%)")
    
    # Trier et limiter
    filtered.sort(key=lambda x: x["score"], reverse=True)
    filtered = filtered[:max_detections]
    
    if not filtered:
        print(f"   ⚠️  Toutes boxes filtrées, fallback SegFormer")
        return segment_floor_auto(image)
    
    print(f"   ✅ {len(filtered)} détection(s) retenue(s)")
    for i, det in enumerate(filtered):
        print(f"      {i+1}. score={det['score']:.2f}, taille={det['box_ratio']*100:.1f}%")
    
    # Étape 2: SAM2
    model, processor = load_sam2()
    boxes = [det["box"] for det in filtered]
    
    inputs = processor(
        images=image,
        input_boxes=[boxes],
        return_tensors="pt"
    ).to("cuda", torch.float16)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    original_sizes = inputs["original_sizes"].tolist()
    masks = processor.post_process_masks(
        outputs.pred_masks,
        original_sizes,
        binarize=False
    )
    
    # Combiner les masques
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(len(boxes)):
        scores = outputs.iou_scores[0][i].cpu().numpy()
        best_idx = scores.argmax()
        mask = masks[0][i][best_idx].cpu().numpy().astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, mask)
    
    mask_image = Image.fromarray(combined_mask, mode="L")
    
    if save_path:
        mask_image.save(save_path)
        print(f"   💾 Masque sauvegardé: {save_path}")
    
    coverage = np.sum(combined_mask > 0) / combined_mask.size * 100
    print(f"   ✅ Segmentation terminée ({coverage:.1f}% de l'image)")
    
    return mask_image
