# Test du nouveau système ADD avec zones spatiales
from segmentation.intent_parser import parse_intent, describe_intent

print("=" * 60)
print("🧪 TEST: NOUVEAU SYSTÈME ADD (NIVEAU PROD)")
print("=" * 60)

# Test 1: ADD flowers
print("\n📝 Test 1: Ajouter des roses")
prompt1 = "Ajouter un peu de roses dans le jardin en premier plan"
intent1 = parse_intent(prompt1)

print(f"   Prompt: {prompt1}")
print(f"   {describe_intent(intent1)}")
print(f"   Action Type: {intent1.action_type} ✅" if intent1.action_type == "ADD" else f"   Action Type: {intent1.action_type} ❌")
print(f"   Object to Add: {intent1.object_to_add}")
print(f"   Location: {intent1.location}")

# Test 2: MODIFY floor
print("\n📝 Test 2: Changer le sol")
prompt2 = "Change the floor to marble"
intent2 = parse_intent(prompt2)

print(f"   Prompt: {prompt2}")
print(f"   {describe_intent(intent2)}")
print(f"   Action Type: {intent2.action_type} ✅" if intent2.action_type == "MODIFY" else f"   Action Type: {intent2.action_type} ❌")

# Test 3: REMOVE object
print("\n📝 Test 3: Supprimer un objet")
prompt3 = "Remove the sofa"
intent3 = parse_intent(prompt3)

print(f"   Prompt: {prompt3}")
print(f"   {describe_intent(intent3)}")
print(f"   Action Type: {intent3.action_type} ✅" if intent3.action_type == "REMOVE" else f"   Action Type: {intent3.action_type} ❌")

# Test 4: ADD avec location complexe
print("\n📝 Test 4: Ajouter dans une zone spécifique")
prompt4 = "Add some trees in the background of the garden"
intent4 = parse_intent(prompt4)

print(f"   Prompt: {prompt4}")
print(f"   {describe_intent(intent4)}")
print(f"   Action Type: {intent4.action_type} ✅" if intent4.action_type == "ADD" else f"   Action Type: {intent4.action_type} ❌")
print(f"   Object to Add: {intent4.object_to_add}")
print(f"   Location: {intent4.location}")

# Test 5: ADD flowers autre formulation
print("\n📝 Test 5: Placer des fleurs")
prompt5 = "Place colorful flowers near the path"
intent5 = parse_intent(prompt5)

print(f"   Prompt: {prompt5}")
print(f"   {describe_intent(intent5)}")
print(f"   Action Type: {intent5.action_type} ✅" if intent5.action_type == "ADD" else f"   Action Type: {intent5.action_type} ❌")
print(f"   Object to Add: {intent5.object_to_add}")
print(f"   Location: {intent5.location}")

print("\n" + "=" * 60)
print("✅ TESTS TERMINÉS")
print("=" * 60)
