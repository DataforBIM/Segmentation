# Canny + resize
import cv2
import numpy as np
from PIL import Image

def make_canny(image, low=80, high=160):
    img = np.array(image)
    edges = cv2.Canny(img, low, high)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)

def compute_output_size(image, max_size):
    w, h = image.size
    ratio = w / h
    if w >= h:
        w2, h2 = max_size, int(max_size / ratio)
    else:
        h2, w2 = max_size, int(max_size * ratio)

    return max(512, w2 // 8 * 8), max(512, h2 // 8 * 8)
