import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if token is None:
    raise RuntimeError("HF_TOKEN non défini")

login(token=token)
print("✅ Login Hugging Face OK")
