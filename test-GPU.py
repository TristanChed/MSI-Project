import torch
import sys

print(f"DIAGNOSTIC SYSTÈME")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# Vérification spécifique AMD
print(f"ROCm version (HIP): {torch.version.hip}") 

# Vérification détection matériel
is_available = torch.cuda.is_available()
print(f"GPU Détecté (Is Available): {is_available}")

if is_available:
    device_count = torch.cuda.device_count()
    print(f"Nombre de GPU: {device_count}")
    print(f"Nom du GPU 0: {torch.cuda.get_device_name(0)}")
    # Capacité de calcul (Architecture)
    cap = torch.cuda.get_device_capability(0)
    print(f"Capability: {cap}")
else:
    print("ERREUR: PyTorch ne détecte pas de GPU.")