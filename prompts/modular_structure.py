# =====================================================
# MODULAR PROMPT STRUCTURE - SDXL
# prompts/modular_structure.py
# =====================================================
# Structure modulaire pour construction de prompts:
# [SCENE STRUCTURE]
# [SUBJECT]
# [ENVIRONMENT]
# [CAMERA / LENS]
# [LIGHTING]
# [MATERIALS]
# [STYLE]
# [NEGATIVE PROMPT]

# =====================================================
# SCENE STRUCTURE - Structure et composition
# =====================================================

SCENE_STRUCTURES = {
    "interior": (
        "interior architectural space, "
        "indoor room layout, "
        "enclosed architectural environment, "
        "interior perspective, "
        "structured interior composition"
    ),
    "exterior": (
        "exterior architectural view, "
        "outdoor building perspective, "
        "open architectural composition, "
        "building facade view, "
        "external structure"
    ),
    "aerial": (
        "aerial view composition, "
        "bird's eye perspective, "
        "top-down architectural layout, "
        "urban architectural overview, "
        "elevated viewpoint"
    ),
    "landscape": (
        "landscape architectural composition, "
        "wide environmental view, "
        "contextual site layout, "
        "architectural landscape integration"
    ),
    "detail": (
        "architectural detail composition, "
        "close-up architectural element, "
        "focused detail view, "
        "element isolation"
    )
}

# =====================================================
# SUBJECT - Sujet principal
# =====================================================

SUBJECTS = {
    "building": (
        "contemporary building, "
        "modern architectural structure, "
        "designed building, "
        "architectural construction"
    ),
    "facade": (
        "building facade, "
        "architectural front elevation, "
        "exterior wall composition, "
        "facade treatment"
    ),
    "interior_space": (
        "interior architectural space, "
        "designed room, "
        "interior volume, "
        "spatial composition"
    ),
    "urban_block": (
        "urban architectural block, "
        "city building ensemble, "
        "urban fabric, "
        "architectural grouping"
    ),
    "roof": (
        "architectural roof structure, "
        "roof composition, "
        "roofing system, "
        "roof geometry"
    ),
    "courtyard": (
        "architectural courtyard, "
        "interior outdoor space, "
        "enclosed open space, "
        "courtyard composition"
    ),
    "entrance": (
        "building entrance, "
        "architectural entry, "
        "entry composition, "
        "access point design"
    )
}

# =====================================================
# ENVIRONMENT - Contexte et environnement
# =====================================================

ENVIRONMENTS = {
    "urban": (
        "urban environment, "
        "city context, "
        "urban setting, "
        "metropolitan context, "
        "built-up area"
    ),
    "residential": (
        "residential neighborhood, "
        "housing context, "
        "residential area, "
        "domestic setting"
    ),
    "park": (
        "park setting, "
        "green space context, "
        "landscaped environment, "
        "natural surroundings"
    ),
    "street": (
        "street context, "
        "urban street environment, "
        "streetscape setting, "
        "street level context"
    ),
    "plaza": (
        "plaza environment, "
        "public square context, "
        "open urban space, "
        "civic space setting"
    ),
    "isolated": (
        "isolated setting, "
        "minimal context, "
        "clean background, "
        "focused environment"
    ),
    "waterfront": (
        "waterfront setting, "
        "water adjacent context, "
        "riverside environment, "
        "coastal location"
    )
}

# =====================================================
# CAMERA / LENS - Prise de vue et perspective
# =====================================================

CAMERA_SETTINGS = {
    "eye_level": (
        "camera at eye level, "
        "human perspective height, "
        "1.6m viewpoint, "
        "ground level view"
    ),
    "low_angle": (
        "low angle shot, "
        "upward perspective, "
        "ground-up view, "
        "elevated subject viewpoint"
    ),
    "high_angle": (
        "high angle shot, "
        "downward perspective, "
        "elevated camera position, "
        "overhead view"
    ),
    "aerial_orthogonal": (
        "orthogonal aerial view, "
        "perpendicular top-down, "
        "plan view perspective, "
        "90-degree overhead"
    ),
    "aerial_oblique": (
        "oblique aerial view, "
        "angled overhead perspective, "
        "45-degree aerial angle, "
        "three-quarter aerial view"
    ),
    "wide_angle": (
        "wide angle lens, "
        "24mm focal length, "
        "expansive field of view, "
        "wide perspective"
    ),
    "normal_lens": (
        "normal lens, "
        "50mm focal length, "
        "natural perspective, "
        "standard field of view"
    ),
    "telephoto": (
        "telephoto lens, "
        "compressed perspective, "
        "85mm+ focal length, "
        "narrow field of view"
    ),
    "straight_verticals": (
        "straight vertical lines, "
        "corrected perspective, "
        "no distortion, "
        "parallel verticals, "
        "architectural photography correction"
    )
}

# =====================================================
# LIGHTING - Éclairage et atmosphère
# =====================================================

LIGHTING_CONDITIONS = {
    "natural_daylight": (
        "natural daylight, "
        "soft natural illumination, "
        "ambient daylight, "
        "neutral day lighting"
    ),
    "golden_hour": (
        "golden hour lighting, "
        "warm sunset light, "
        "late afternoon illumination, "
        "soft directional sunlight"
    ),
    "overcast": (
        "overcast sky lighting, "
        "diffused cloud light, "
        "soft even illumination, "
        "cloudy day lighting"
    ),
    "blue_hour": (
        "blue hour lighting, "
        "twilight illumination, "
        "dusk atmospheric light, "
        "evening ambient light"
    ),
    "bright_sun": (
        "bright sunlight, "
        "strong directional sun, "
        "high contrast lighting, "
        "direct solar illumination"
    ),
    "soft_shadows": (
        "soft shadow definition, "
        "gentle shadow gradients, "
        "natural shadow transitions, "
        "realistic shadow behavior"
    ),
    "hard_shadows": (
        "hard shadow definition, "
        "sharp shadow edges, "
        "strong shadow contrast, "
        "direct light shadows"
    ),
    "neutral_lighting": (
        "neutral even lighting, "
        "balanced illumination, "
        "no color cast, "
        "natural light temperature"
    )
}

# =====================================================
# MATERIALS - Matériaux et textures
# =====================================================

MATERIALS = {
    "concrete": (
        "concrete material, "
        "concrete surfaces, "
        "raw concrete texture, "
        "concrete finish"
    ),
    "brick": (
        "brick material, "
        "masonry surfaces, "
        "brick texture, "
        "brick facade"
    ),
    "glass": (
        "glass material, "
        "glazed surfaces, "
        "transparent glazing, "
        "glass facade elements"
    ),
    "wood": (
        "wood material, "
        "timber surfaces, "
        "wood texture, "
        "wooden elements"
    ),
    "metal": (
        "metal material, "
        "metallic surfaces, "
        "metal cladding, "
        "metal finishes"
    ),
    "stone": (
        "stone material, "
        "natural stone surfaces, "
        "stone texture, "
        "stone finish"
    ),
    "plaster": (
        "plaster material, "
        "rendered surfaces, "
        "smooth plaster finish, "
        "stucco texture"
    ),
    "mixed_materials": (
        "mixed material palette, "
        "varied surface materials, "
        "material diversity, "
        "multiple textures"
    ),
    "realistic_weathering": (
        "realistic material weathering, "
        "natural aging, "
        "authentic patina, "
        "real-world wear"
    ),
    "clean_surfaces": (
        "clean material surfaces, "
        "well-maintained appearance, "
        "fresh finish, "
        "minimal weathering"
    )
}

# =====================================================
# STYLE - Style photographique et traitement
# =====================================================

STYLES = {
    "photorealistic": (
        "photorealistic, "
        "raw photograph, "
        "unprocessed photo, "
        "real camera capture, "
        "authentic photography, "
        "genuine real-world scene"
    ),
    "architectural_photo": (
        "professional architectural photography, "
        "architectural documentation, "
        "building photography, "
        "architectural image capture"
    ),
    "high_quality": (
        "8k resolution, "
        "high definition, "
        "professional quality, "
        "sharp details, "
        "crisp image"
    ),
    "natural_colors": (
        "natural realistic colors, "
        "accurate color reproduction, "
        "true-to-life colors, "
        "neutral color grading"
    ),
    "minimal_processing": (
        "minimal post-processing, "
        "original image preservation, "
        "authentic appearance, "
        "unfiltered look"
    ),
    "documentary": (
        "documentary style, "
        "factual representation, "
        "objective photography, "
        "truthful capture"
    ),
    "clean_composition": (
        "clean composition, "
        "organized frame, "
        "balanced elements, "
        "clear visual hierarchy"
    )
}

# =====================================================
# NEGATIVE PROMPT - Éléments à éviter
# =====================================================

NEGATIVE_BASE = (
    "artifacts, visual artifacts, noise artifacts, jpeg artifacts, compression artifacts, "
    "glitches, visual glitches, rendering errors, distortion, "
    "haloing, edge artifacts, banding, posterization"
)

NEGATIVE_RENDERING = (
    "3d render, 3d model, cgi, computer generated imagery, "
    "video game graphics, game engine, gaming render, unreal engine, unity, "
    "synthetic, artificial render, digital render"
)

NEGATIVE_ARTISTIC = (
    "cartoon, anime, illustration, painting, drawing, comic, sketch, "
    "stylized, artistic style, art filter, painted effect, "
    "cel shading, flat shading, toon shading, posterized"
)

NEGATIVE_MATERIALS = (
    "plastic look, toy appearance, miniature effect, diorama, model, "
    "fake textures, synthetic textures, unnatural textures, "
    "perfect surfaces, too clean, overly processed"
)

NEGATIVE_COLOR = (
    "yellow tint, yellow cast, orange tint, sepia tone, color cast, wrong colors, "
    "enhanced saturation, boosted colors, vibrant unrealistic colors, "
    "artificial colors, processed colors"
)

NEGATIVE_LIGHTING = (
    "dramatic lighting, studio lights, artificial lighting effects, "
    "unrealistic lighting, fake shadows, wrong lighting direction"
)

NEGATIVE_QUALITY = (
    "low quality, low resolution, blurry, soft focus, pixelated, "
    "noise, grain, overexposed, underexposed"
)

NEGATIVE_GEOMETRY = (
    "distorted geometry, warped surfaces, bent lines, curved verticals, "
    "impossible architecture, wrong perspective, broken geometry, "
    "melted surfaces, stretched textures"
)

NEGATIVE_EXTRAS = (
    "text, watermark, logo, UI, HUD, interface elements, "
    "added objects, extra elements, hallucinated content"
)

def get_full_negative_prompt() -> str:
    """Retourne le prompt négatif complet"""
    return (
        f"{NEGATIVE_BASE}, "
        f"{NEGATIVE_RENDERING}, "
        f"{NEGATIVE_ARTISTIC}, "
        f"{NEGATIVE_MATERIALS}, "
        f"{NEGATIVE_COLOR}, "
        f"{NEGATIVE_LIGHTING}, "
        f"{NEGATIVE_QUALITY}, "
        f"{NEGATIVE_GEOMETRY}, "
        f"{NEGATIVE_EXTRAS}"
    )
