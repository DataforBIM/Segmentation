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
        "vue aérienne architecturale, "
        "aerial architectural photography, "
        "drone view, "
        "oblique aerial perspective, "
        "large scale context visible"
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
        "added lights, extra lights, "
        "added decorations, extra decorations, "
        "modified walls, changed walls, "
        "modified floor, changed floor, "
        "modified ceiling, changed ceiling, "
        "removed objects, missing objects, "
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
        "cartoon, illustration, anime, painting, sketch, 3d render, cgi, "
        "blurry, low quality, noise, artifacts, "
        "distorted perspective, impossible angle, "
        "warped buildings, curved straight lines, "
        "unrealistic scale, wrong proportions, "
        "floating structures, disconnected roads, "
        "fake vegetation, plastic trees, "
        "unnatural patterns, grid distortion, "
        "added buildings, removed structures, "
        "modified urban layout, "
        "text, watermark, logo, people, cars"
    )
}

