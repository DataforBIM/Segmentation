from prompts.prompt_layers import PromptLayer
from intent.intent_config import IntentConfig
from prompts.modular_builder import build_modular_prompt

def build_prompt_layers(
    prompt_config,
    intent: IntentConfig
) -> list[PromptLayer]:
    """
    Transforme PromptConfig + IntentConfig en 3 couches GPT-like
    
    LAYER A — CORE      (Quoi + Où)
    LAYER B — CONTEXT   (Contraintes + Intégration)
    LAYER C — QUALITY   (Garde-fous visuels)
    """

    layers: list[PromptLayer] = []

    # ==============================
    # 🔹 LAYER A — CORE (Quoi + Où)
    # ==============================
    # Contient:
    # - type de scène
    # - sujet dominant
    # - action principale (add / modify / enhance)
    # - cible principale
    
    base_prompt, _ = build_modular_prompt(prompt_config)
    
    core_parts = [base_prompt]
    
    # Action principale + cible
    if intent.action == "add":
        core_parts.append(f"add {intent.target}")
        if intent.location:
            core_parts.append(f"in {intent.location}")
    
    elif intent.action == "modify":
        core_parts.append(f"modify {intent.target}")
        if intent.location:
            core_parts.append(f"at {intent.location}")
    
    elif intent.action == "enhance":
        core_parts.append(f"enhance {intent.target}")
    
    elif intent.action == "replace":
        core_parts.append(f"replace {intent.target}")
        if intent.location:
            core_parts.append(f"at {intent.location}")
    
    layers.append(PromptLayer(
        role="core",
        text=", ".join(core_parts),
        strength="high"
    ))

    # ==============================
    # 🔹 LAYER B — CONTEXT (Contraintes + Intégration)
    # ==============================
    # Contient:
    # - contraintes utilisateur
    # - contraintes image
    # - règles de préservation
    # - intégration douce
    # - règles locales
    
    context_parts = []
    
    # Contraintes utilisateur (priorité maximale)
    for constraint in intent.constraints:
        context_parts.append(constraint)
    
    # Règles d'intégration douce
    context_parts.extend([
        "naturally blending with surrounding environment",
        "seamless integration",
        "no sharp edges or discontinuities",
        "preserving existing lighting and atmosphere",
        "matching perspective and scale"
    ])
    
    layers.append(PromptLayer(
        role="context",
        text=", ".join(context_parts),
        strength="high"
    ))

    # ==============================
    # 🔹 LAYER C — QUALITY (Garde-fous visuels)
    # ==============================
    # Contient:
    # - réalisme
    # - imperfections
    # - anti-IA look
    # - neutralité colorimétrique
    
    quality_parts = [
        "photographic realism",
        "subtle imperfections",
        "natural wear and aging",
        "realistic textures",
        "no artificial look",
        "no oversaturation",
        "neutral color grading",
        "authentic materials behavior"
    ]
    
    layers.append(PromptLayer(
        role="quality",
        text=", ".join(quality_parts),
        strength="medium"
    ))

    return layers
