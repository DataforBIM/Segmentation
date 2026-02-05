from intent.intent_parser import parse_intent
from intent.intent_config import IntentConfig
from prompts.prompt_layer_builder import build_prompt_layers
from prompts.builders import build_prompts

def build_gptlike_prompt(user_prompt: str):
    # 1. Intent GPT
    intent_data = parse_intent(user_prompt)
    intent = IntentConfig(**intent_data)

    # 2. PromptConfig existant (TON code)
    prompt_positive, prompt_negative = build_prompts(
        user_prompt=user_prompt,
        auto_detect=True
    )

    # On reconstruit PromptConfig depuis ton builder
    # (ou tu peux l’exposer directement)
    from prompts.modular_builder import auto_detect_config_from_prompt
    prompt_config = auto_detect_config_from_prompt(user_prompt)

    # 3. Prompt Layering
    layers = build_prompt_layers(prompt_config, intent)

    # 4. Rendu final
    final_prompt = ", ".join(layer.render() for layer in layers)

    return final_prompt
