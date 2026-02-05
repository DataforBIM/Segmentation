# Test de la classification sémantique (basée sur intention)
from segmentation.intent_parser import parse_intent, describe_intent

print("=" * 70)
print("🧠 TEST: CLASSIFICATION SÉMANTIQUE (BASÉE SUR L'INTENTION)")
print("=" * 70)
print()

# ============================================
# ADD: Introduction de nouveaux éléments
# ============================================
print("🌱 ACTIONS ADD (Introduire quelque chose de nouveau)")
print("-" * 70)

add_prompts = [
    "Ajouter un peu de roses dans le jardin en premier plan",
    "Mettre quelques fleurs dans le jardin",  # ✨ Pas de verbe "ajouter"
    "Le jardin avec des roses colorées",  # ✨ Pas de verbe du tout
    "Un peu de végétation sur le côté gauche",
    "Quelques arbres dans le fond",
    "Des plantes décoratives près de l'entrée"
]

for prompt in add_prompts:
    intent = parse_intent(prompt)
    status = "✅" if intent.action_type == "ADD" else "❌"
    print(f"{status} {intent.action_type:6} | {prompt}")
    if intent.action_type == "ADD":
        print(f"          → Object: {intent.object_to_add}, Location: {intent.location}")

print()

# ============================================
# MODIFY: Changement de propriété existante
# ============================================
print("🔄 ACTIONS MODIFY (Modifier ce qui existe)")
print("-" * 70)

modify_prompts = [
    "Change the floor to marble",
    "Changer la couleur du mur en blanc",
    "Le sol en marbre",  # ✨ Transformation implicite
    "La façade en verre moderne",  # ✨ Pas de verbe "changer"
    "Transformer le plafond en bois",
    "Le mur blanc",  # ✨ Changement de couleur implicite
    "Make the walls blue"
]

for prompt in modify_prompts:
    intent = parse_intent(prompt)
    status = "✅" if intent.action_type == "MODIFY" else "❌"
    print(f"{status} {intent.action_type:6} | {prompt}")

print()

# ============================================
# REMOVE: Suppression d'éléments
# ============================================
print("🗑️  ACTIONS REMOVE (Supprimer)")
print("-" * 70)

remove_prompts = [
    "Remove the sofa",
    "Supprimer la table",
    "Enlever les meubles",
    "Delete the car",
    "Effacer l'arbre"
]

for prompt in remove_prompts:
    intent = parse_intent(prompt)
    status = "✅" if intent.action_type == "REMOVE" else "❌"
    print(f"{status} {intent.action_type:6} | {prompt}")

print()

# ============================================
# CAS LIMITES
# ============================================
print("🔍 CAS LIMITES (Ambigus)")
print("-" * 70)

edge_cases = [
    ("Replace the floor with marble", "MODIFY", "Remplacement = modification"),
    ("Add modern style to the facade", "MODIFY", "Style = propriété"),
    ("Des roses dans le jardin au lieu du gazon", "ADD", "Ajout avec contexte"),
    ("Améliorer la façade", "MODIFY", "Amélioration = modification")
]

for prompt, expected, comment in edge_cases:
    intent = parse_intent(prompt)
    status = "✅" if intent.action_type == expected else f"❌ (got {intent.action_type})"
    print(f"{status} {expected:6} | {prompt}")
    print(f"          → {comment}")

print()
print("=" * 70)
print("✅ TESTS TERMINÉS")
print("=" * 70)
