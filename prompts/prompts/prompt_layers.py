class PromptLayer:
    """
    Une couche logique du prompt (GPT-like)
    
    3 couches stratégiques:
    - LAYER A (CORE): Quoi + Où
    - LAYER B (CONTEXT): Contraintes + Intégration  
    - LAYER C (QUALITY): Garde-fous visuels
    """

    def __init__(
        self,
        role: str,
        text: str,
        strength: str = "medium"
    ):
        self.role = role          # core | context | quality
        self.text = text
        self.strength = strength  # low | medium | high

    def render(self) -> str:
        """
        Rend la couche sous forme de texte
        avec pondération implicite
        """
        if self.strength == "low":
            return f"subtle {self.text}"
        if self.strength == "high":
            return f"strongly {self.text}"
        return self.text

    def __repr__(self):
        return f"PromptLayer(role={self.role}, strength={self.strength})"
