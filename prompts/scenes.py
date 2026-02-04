# prompts/scenes.py

SCENE_PROMPTS = {
    "INTERIOR": (
        "architecture intérieure contemporaine, "
        "interior architectural photography, "
        "wide shot interior, "
        "camera at eye level, "
        "straight verticals, "
        "realistic room proportions"
    ),
    "EXTERIOR": (
        "architecture contemporaine extérieure, "
        "exterior architectural photography, "
        "wide shot exterior, "
        "building fully visible, "
        "camera at eye level, "
        "straight verticals, "
        "realistic scale and proportions"
    ),
    "AERIAL": (
        "architectural aerial reconstruction, "
        "clean geometry, "
        "straight walls, "
        "realistic roof tiles, "
        "urban european residential block, "
        "accurate proportions, "
        "neutral lighting, "
        "no photogrammetry artifacts, "
        "sharp edges, "
        "high structural clarity, "
        "raw aerial photograph, "
        "real drone photo, "
        "natural colors, "
        "authentic lighting, "
        "realistic details preservation, "
        "photographic quality"
    )
}

NEGATIVE_PROMPTS = {
    "INTERIOR": (
        "cartoon, illustration, anime, painting, sketch, 3d render, cgi, "
        "blurry, low quality, noise, artifacts, "
        "warped walls, curved walls, distorted perspective, "
        "broken geometry, impossible room layout, "
        "wrong ceiling height, disproportionate room, "
        "added objects, extra objects, "
        "added furniture, extra furniture, "
        "modified walls, changed walls, "
        "modified ceiling, changed ceiling, "
        "multiple heads, extra limbs, "
        "fused fingers, six fingers, "
        "no face, faceless, "
        "text, watermark, logo"
    ),

    "EXTERIOR": (
        "cartoon, illustration, anime, painting, sketch, 3d render, cgi, "
        "blurry, low quality, noise, artifacts, "
        "distorted building, impossible architecture, "
        "warped façade, broken perspective, "
        "tilted verticals, "
        "unrealistic scale, giant trees, tiny cars, "
        "floating buildings, disconnected structure, "
        "fake trees, plastic plants, "
        "unnatural sky, fake clouds, "
        "added windows, removed floors, "
        "changed materials, "
        "extra buildings, modified landscape, "
        "text, watermark, logo"
    ),

    "AERIAL": (
        "yellow tint, yellow cast, yellow buildings, orange tint, warm color cast, "
        "artificial colors, wrong colors, color grading, color filter, "
        "video game, game graphics, gaming render, unreal engine, unity engine, "
        "3d render, cgi render, computer generated, digital render, synthetic, "
        "cartoon, anime, illustration, painting, drawing, sketch, comic, "
        "stylized, painted effect, art filter, processed look, "
        "miniature effect, tilt-shift, toy town, model city, "
        "plastic materials, shiny surfaces, glossy finish, "
        "oversaturated, boosted saturation, enhanced colors, vibrant colors, "
        "dramatic lighting, artificial lights, game lighting, "
        "blurry, low quality, pixelated, unrealistic"
    )
}

