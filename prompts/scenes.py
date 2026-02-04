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
        "urban architecture reconstruction from aerial view, "
        "clean and corrected geometry, "
        "straightened walls and façades, "
        "preserve existing windows and architectural details, "
        "regular window openings, "
        "consistent facade fenestration, "
        "preserve existing window positions, "
        "architectural openings clearly defined, "
        "accurate building volumes, "
        "realistic roof shapes and tiles, "
        "european residential urban block, "
        "true-to-scale proportions, "
        "orthogonal architecture, "
        "structural clarity over texture fidelity, "
        "corrected photogrammetry, "
        "no surface warping, "
        "neutral daylight illumination, "
        "physically plausible materials, "
        "natural unprocessed colors, "
        "high realism, "
        "architectural accuracy"
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
        "removed floors, "
        "changed materials, "
        "extra buildings, modified landscape, "
        "text, watermark, logo"
    ),

    "AERIAL": (
        "photogrammetry artifacts, mesh distortion, melted geometry, "
        "warped surfaces, stretched textures, triangulated surfaces, "
        "low-poly geometry, decimated mesh, "
        "facade bending, roof warping, collapsed edges, "
        "floating surfaces, geometry seams, "
        "yellow tint, yellow cast, orange cast, warm color grading, "
        "artificial colors, color filter, processed look, "
        "video game, game graphics, unreal engine, unity engine, "
        "3d render, cgi, synthetic, digital render, "
        "cartoon, anime, illustration, painting, sketch, "
        "miniature effect, tilt-shift, toy city, "
        "plastic materials, glossy surfaces, "
        "oversaturated colors, exaggerated contrast, "
        "dramatic lighting, cinematic lighting, "
        "blurry, low resolution, pixelated, "
        "hallucinated buildings, added structures, "
        "changed urban layout, text, watermark, logo"
    )
}

