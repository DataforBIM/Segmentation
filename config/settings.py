# Flags & constantes
USE_SDXL = True
USE_REFINER = True
USE_UPSCALER = True
USE_SEGMENTATION = True  # Activer SAM2/SegFormer

SDXL_MODEL = "SG161222/RealVisXL_V4.0"
CONTROLNET_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Segmentation settings
SEGMENT_TARGET = "floor"  # "floor", "wall", "ceiling", "custom"
SEGMENT_METHOD = "auto"   # "auto", "points", "box"
SEGMENT_DILATE = 5        # Dilatation du masque (pixels)
SEGMENT_FEATHER = 10      # Adoucissement des bords

# Generation mode
USE_INPAINTING = False    # False = ControlNet + masque (cumule les deux !)

SEED = 123456
MAX_SIZE = 1024
