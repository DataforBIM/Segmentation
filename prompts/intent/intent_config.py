class IntentConfig:
    """
    Intention opérationnelle extraite du prompt utilisateur
    """
    def __init__(
        self,
        action: str,
        target: str,
        location: str = None,
        constraints: list[str] = None,
        attributes: dict = None
    ):
        self.action = action            # add | modify | enhance | replace
        self.target = target            # flowers | facade | lighting
        self.location = location        # garden | foreground
        self.constraints = constraints or []
        self.attributes = attributes or {}
