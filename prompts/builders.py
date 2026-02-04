# prompts/builder.py

from prompts.base import BASE_PROMPT, BASE_NEGATIVE
from prompts.scenes import SCENE_PROMPTS, NEGATIVE_PROMPTS

def build_prompts(
    scene_type: str,
    user_prompt: str,
    aerial_elements: list[str] = None
) -> tuple[str, str]:
    """
    Retourne (prompt, negative_prompt)
    
    Args:
        scene_type: Type de scène (INTERIOR, EXTERIOR, AERIAL, etc.)
        user_prompt: Prompt de l'utilisateur
        aerial_elements: Liste des éléments détectés pour les scènes aériennes
    
    Returns:
        (prompt_positif, prompt_négatif)
    """
    
    # CAS SPÉCIAL: Scène aérienne avec éléments détectés
    if scene_type == "AERIAL" and aerial_elements:
        from prompts.aerial_elements import build_aerial_prompt, get_element_description
        
        # Construire des prompts spécifiques pour chaque élément
        aerial_positive, aerial_negative = build_aerial_prompt(user_prompt, aerial_elements)
        
        # Ajouter les prompts de scène
        scene_prompt = SCENE_PROMPTS.get(scene_type, "")
        scene_negative = NEGATIVE_PROMPTS.get(scene_type, "")
        
        final_prompt = (
            f"{aerial_positive}, "
            f"{BASE_PROMPT}, "
            f"{scene_prompt}"
        )
        
        final_negative = (
            f"{aerial_negative}, "
            f"{scene_negative}, "
            f"{BASE_NEGATIVE}"
        )
        
        # Afficher les éléments détectés
        elements_str = ", ".join([get_element_description(e) for e in aerial_elements])
        print(f"   🎯 Prompts enrichis pour éléments aériens: {elements_str}")
        
        return final_prompt, final_negative
    
    # CAS STANDARD: Autres scènes
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
