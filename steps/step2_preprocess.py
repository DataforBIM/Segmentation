# Canny + resize
import cv2
import numpy as np
from PIL import Image

def make_canny(image: Image.Image, save_path: str = None) -> Image.Image:
    """
    Applique l'algorithme Canny pour détecter les contours dans l'image.

    Args:
        image: Image PIL d'entrée
        save_path: Chemin pour sauvegarder l'image des contours (optionnel)

    Returns:
        Image PIL avec les contours détectés
    """
    import cv2
    import numpy as np

    print("   🔍 Génération des contours avec Canny...")

    # Convertir PIL → numpy
    img_np = np.array(image.convert("L"))

    # Appliquer Canny
    edges = cv2.Canny(img_np, 100, 200)

    # Convertir numpy → PIL
    edges_image = Image.fromarray(edges)

    if save_path:
        edges_image.save(save_path)
        print(f"   💾 Contours sauvegardés: {save_path}")

    return edges_image

def compute_output_size(image, max_size):
    w, h = image.size
    ratio = w / h
    if w >= h:
        w2, h2 = max_size, int(max_size / ratio)
    else:
        h2, w2 = max_size, int(max_size * ratio)

    return max(512, w2 // 8 * 8), max(512, h2 // 8 * 8)
