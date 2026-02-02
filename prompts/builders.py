# prompts/builder.py

from prompts.base import BASE_PROMPT, BASE_NEGATIVE
from prompts.scenes import SCENE_PROMPTS, NEGATIVE_PROMPTS

def build_prompts(
    scene_type: str,
    user_prompt: str
) -> tuple[str, str]:
    """
    Retourne (prompt, negative_prompt)
    """

    scene_prompt = SCENE_PROMPTS.get(scene_type, "")
    scene_negative = NEGATIVE_PROMPTS.get(scene_type, "")

    final_prompt = (
        f"{user_prompt}, "
        f"{BASE_PROMPT}, "
        f"{scene_prompt}"
    )

    final_negative = (
        f"{scene_negative}, "
        f"{BASE_NEGATIVE}"
    )

    return final_prompt, final_negative
