# Canny + resize + Depth + OpenPose + Normal
import cv2
import numpy as np
from PIL import Image


def make_canny(image: Image.Image, save_path: str = None, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """
    Applique l'algorithme Canny pour détecter les contours dans l'image.

    Args:
        image: Image PIL d'entrée
        save_path: Chemin pour sauvegarder l'image des contours (optionnel)
        low_threshold: Seuil bas pour Canny (défaut: 100, pour soft aerial: 30)
        high_threshold: Seuil haut pour Canny (défaut: 200, pour soft aerial: 80)

    Returns:
        Image PIL avec les contours détectés
    """
    print(f"   🔍 Génération des contours avec Canny (soft: {low_threshold}/{high_threshold})...")

    # Convertir PIL → numpy
    img_np = np.array(image.convert("L"))

    # Appliquer Canny avec seuils personnalisés
    edges = cv2.Canny(img_np, low_threshold, high_threshold)

    # Convertir numpy → PIL
    edges_image = Image.fromarray(edges)

    if save_path:
        edges_image.save(save_path)
        print(f"   💾 Contours sauvegardés: {save_path}")

    return edges_image


def make_depth(image: Image.Image, save_path: str = None) -> Image.Image:
    """
    Génère une carte de profondeur à partir de l'image.
    
    Args:
        image: Image PIL d'entrée
        save_path: Chemin pour sauvegarder la carte de profondeur (optionnel)
    
    Returns:
        Image PIL avec la carte de profondeur
    """
    from transformers import pipeline as hf_pipeline
    
    print("   🏔️  Génération de la carte de profondeur...")
    
    # Charger le modèle de depth estimation
    depth_estimator = hf_pipeline("depth-estimation", model="Intel/dpt-large")
    
    # Générer la carte de profondeur
    depth_result = depth_estimator(image)
    depth_image = depth_result["depth"]
    
    if save_path:
        depth_image.save(save_path)
        print(f"   💾 Profondeur sauvegardée: {save_path}")
    
    return depth_image


def make_openpose(image: Image.Image, save_path: str = None) -> Image.Image:
    """
    Détecte les poses humaines avec OpenPose.
    
    Args:
        image: Image PIL d'entrée
        save_path: Chemin pour sauvegarder l'image OpenPose (optionnel)
    
    Returns:
        Image PIL avec les poses détectées
    """
    from controlnet_aux import OpenposeDetector
    
    print("   🧍 Génération des poses avec OpenPose...")
    
    # Charger le détecteur OpenPose
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    # Détecter les poses
    pose_image = openpose(image)
    
    if save_path:
        pose_image.save(save_path)
        print(f"   💾 Poses sauvegardées: {save_path}")
    
    return pose_image


def make_normal(image: Image.Image, save_path: str = None) -> Image.Image:
    """
    Génère une carte de normales à partir de l'image.
    
    Args:
        image: Image PIL d'entrée
        save_path: Chemin pour sauvegarder la carte de normales (optionnel)
    
    Returns:
        Image PIL avec la carte de normales
    """
    from transformers import pipeline as hf_pipeline
    
    print("   🔷 Génération de la carte de normales...")
    
    # Utiliser depth pour générer des normales approximatives
    depth_estimator = hf_pipeline("depth-estimation", model="Intel/dpt-large")
    depth_result = depth_estimator(image)
    depth_array = np.array(depth_result["depth"])
    
    # Calculer les gradients pour approximer les normales
    grad_x = cv2.Sobel(depth_array, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_array, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normaliser et convertir en normales RGB
    normal = np.dstack((-grad_x, -grad_y, np.ones_like(depth_array)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)
    normal = ((normal + 1) * 127.5).astype(np.uint8)
    
    normal_image = Image.fromarray(normal)
    
    if save_path:
        normal_image.save(save_path)
        print(f"   💾 Normales sauvegardées: {save_path}")
    
    return normal_image


def compute_output_size(image, max_size):
    w, h = image.size
    ratio = w / h
    if w >= h:
        w2, h2 = max_size, int(max_size / ratio)
    else:
        h2, w2 = max_size, int(max_size * ratio)

    return max(512, w2 // 8 * 8), max(512, h2 // 8 * 8)
