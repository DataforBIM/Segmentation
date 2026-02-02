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

    if any(w in caption for w in ["aerial", "drone", "top view", "bird"]):
        return "AERIAL"
    if any(w in caption for w in ["room", "interior", "bedroom", "living"]):
        return "INTERIOR"
    return "EXTERIOR"
