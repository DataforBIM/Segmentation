from openai import OpenAI
import json

client = OpenAI()

SYSTEM_PROMPT = """
You are an intent extraction engine.

Extract structured information ONLY.
Do NOT invent new objects.
Do NOT add artistic style.

Return valid JSON with:
- action: add | modify | enhance | replace
- target
- location (optional)
- constraints (list)
"""

def parse_intent(user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)
