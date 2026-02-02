# Variables d'environnement
import os
from pathlib import Path
import cloudinary
import cloudinary.uploader


# =====================================================
# CHARGEMENT DU FICHIER .env (si disponible)
# =====================================================
def load_env_file():
    """Charge les variables depuis le fichier .env"""
    env_path = Path(__file__).parent.parent / ".env"
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Fichier .env chargé")
    else:
        print("⚠️  Fichier .env non trouvé, utilisation des variables système")


# Charger automatiquement au démarrage
load_env_file()


def get_env(name: str) -> str:
    """Récupère une variable d'environnement avec validation"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"❌ Variable d'environnement manquante : {name}")
    return value


# =====================================================
# CLOUDINARY
# =====================================================
cloudinary.config(
    cloud_name=get_env("CLOUDINARY_CLOUD_NAME"),
    api_key=get_env("CLOUDINARY_API_KEY"),
    api_secret=get_env("CLOUDINARY_API_SECRET"),
    secure=True
)

print("✅ Cloudinary configuré")


# =====================================================
# HUGGING FACE TOKEN (optionnel pour modèles privés)
# =====================================================
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")  # Optionnel

if HF_TOKEN:
    print("✅ Hugging Face token détecté")
else:
    print("⚠️  Hugging Face token non trouvé (OK pour modèles publics)")
