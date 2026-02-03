# BASE_PROMPT / BASE_NEGATIVE
# prompts/base.py

BASE_PROMPT = (
    "ONLY transform floor surface into luxurious white marble flooring, "
    "polished marble tiles, high-end marble texture, "
    "realistic marble veining, premium marble finish, "
    "STRICTLY preserve all walls exactly as original, "
    "STRICTLY preserve all ceiling exactly as original, "
    "STRICTLY preserve all lights and lamps exactly as original, "
    "STRICTLY preserve all furniture exactly as original, "
    "do not add any new elements, do not remove any elements, "
)

BASE_NEGATIVE = (
    "different wall texture, changed wall color, modified walls, "
    "different ceiling, modified ceiling, added ceiling elements, "
    "missing lights, removed lamps, missing lamps, added lights, extra lights, "
    "missing furniture, removed furniture, added furniture, extra furniture, "
    "missing objects, removed objects, added objects, extra objects, "
    "overexposed, underexposed, flat lighting, "
    "compression artifacts, pixelated"
)
