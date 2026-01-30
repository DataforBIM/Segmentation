cat << 'EOF' > login_hf.py
from huggingface_hub import login

print("🔐 Hugging Face login")
print("➡️ Colle ton token HF quand demandé")
login()
print("✅ Login réussi")
EOF
