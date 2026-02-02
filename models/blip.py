# Scene detection
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def detect_scene_type(image):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(output[0], skip_special_tokens=True).lower()

    print(f"   📝 Description BLIP: '{caption}'")

    # Détection de vues aériennes
    if any(w in caption for w in ["aerial", "drone", "top view", "bird", "bird's eye", "overhead", "from above", "rooftop", "roof", "google earth"]):
        return "AERIAL"
    
    # Détection d'intérieurs
    if any(w in caption for w in ["room", "interior", "bedroom", "living", "kitchen", "bathroom", "indoor"]):
        return "INTERIOR"
    
    # Détection d'extérieurs architecturaux
    if any(w in caption for w in ["building", "architecture", "city", "urban", "structure"]):
        return "EXTERIOR"
    
    # Détection de personnes/humains
    if any(w in caption for w in ["person", "people", "man", "woman", "human", "child", "boy", "girl", "crowd", "portrait", "face"]):
        return "HUMAN"
    
    # Détection d'animaux
    if any(w in caption for w in ["cat", "dog", "animal", "bird", "horse", "pet", "wildlife", "fish", "lion", "tiger", "elephant", "bear"]):
        return "ANIMAL"
    
    # Détection de produits/objets
    if any(w in caption for w in ["product", "object", "item", "bottle", "phone", "laptop", "car", "vehicle", "furniture", "tool", "device", "gadget"]):
        return "PRODUCT"
    
    return "GENERIC"
