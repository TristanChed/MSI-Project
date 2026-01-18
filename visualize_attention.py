import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from torchvision import transforms

# On importe tes outils existants
from utils import (
    load_model_for_eval, RogerViolletDataset,
    PATH_FINETUNED_1, PATH_FINETUNED_2, PATH_SCRATCH, 
    DATASET_PATH, DEVICE
)

# ================= CONFIGURATION =================
TARGET_IMAGE_NAME = "467"  # L'image à analyser
IMAGE_SIZE_DISPLAY = (1024, 1024) # Taille pour l'affichage final
THRESHOLD = 0.6 # Pour nettoyer l'attention (garder les 60% les plus forts)
# =================================================

class AttentionRecorder:
    """
    Classe utilitaire pour capturer les poids d'attention.
    Gère les différences d'architecture entre DINOv2 (Hub) et Torchvision (Scratch).
    """
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        self.attentions = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # On attache un 'hook' sur la dernière couche d'attention
        if self.model_type in ["dinov2", "random"]:
            # Architecture DINOv2 (Facebook)
            # La couche est généralement : model.blocks[-1].attn
            target_layer = self.model.blocks[-1].attn
            self.hooks.append(target_layer.register_forward_hook(self._hook_dinov2))
            
        elif self.model_type == "scratch":
            # Architecture Torchvision (Wrapper)
            # La couche est : model.backbone.encoder.layers[-1].self_attention
            target_layer = self.model.backbone.encoder.layers[-1].self_attention
            self.hooks.append(target_layer.register_forward_hook(self._hook_torchvision))

    def _hook_dinov2(self, module, input, output):
        # DINOv2 Hub code: input[0] est x. 
        # On doit recalculer Q, K car DINOv2 utilise souvent MemEffAttention (optimisé)
        # qui ne retourne pas la matrice d'attention brute.
        x = input[0]
        B, N, C = x.shape
        # module.qkv est une couche Linear
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        self.attentions.append(attn.detach().cpu())

    def _hook_torchvision(self, module, input, output):
        # Torchvision MultiheadAttention
        x = input[0] # (Sequence, Batch, Features) ou (Batch, Seq, Feat) selon config
        
        # Torchvision gère parfois batch_first=False
        if x.dim() == 3:
            if self.model.backbone.encoder.layers[-1].self_attention.batch_first:
                pass # (B, N, C)
            else:
                x = x.transpose(0, 1) # (N, B, C) -> (B, N, C)
        
        B, N, C = x.shape
        num_heads = module.num_heads
        head_dim = C // num_heads
        
        # Projection Q, K, V (c'est un gros Linear combiné in_proj)
        qkv = torch.nn.functional.linear(x, module.in_proj_weight, module.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        self.attentions.append(attn.detach().cpu())

    def get_last_attention(self):
        if not self.attentions: return None
        return self.attentions[-1] # Retourne (B, Heads, N, N)

    def close(self):
        for h in self.hooks: h.remove()


def visualize_attention(img_path, model, model_type, img_size):
    # 1. Préparation Image
    # Note: On garde l'image originale pour l'affichage
    original_img = Image.open(img_path).convert('RGB')
    
    # Transform pour le modèle
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(original_img).unsqueeze(0).to(DEVICE)

    # 2. Enregistrement et Forward
    recorder = AttentionRecorder(model, model_type)
    
    with torch.no_grad():
        if model_type in ["dinov2", "random"]:
            model.forward_features(img_tensor)
        else:
            model(img_tensor)
            
    # 3. Récupération Attention
    # Shape: (1, 6, N_tokens, N_tokens) où 6 est le nombre de têtes
    attn_map = recorder.get_last_attention()
    recorder.close()
    
    if attn_map is None: return original_img # Fail safe
    
    # 4. Traitement de la Map
    # On prend l'attention du CLS token (index 0) vers tous les patchs (1:)
    # On fait la moyenne sur toutes les têtes (dim 1)
    # attn_map[0, :, 0, 1:] -> (6, N_patches)
    cls_attn = attn_map[0, :, 0, 1:].mean(0) # (N_patches,)
    
    # Reshape en grille 2D
    # N_patches = H * W. Pour 252px patch 14 -> 18*18 = 324
    # Pour 256px patch 16 -> 16*16 = 256
    n_patches = cls_attn.shape[0]
    grid_size = int(np.sqrt(n_patches))
    cls_attn = cls_attn.reshape(grid_size, grid_size)
    
    # Normalisation pour l'affichage
    cls_attn = cls_attn.numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    
    # Upscale de la grille d'attention vers la taille de l'image
    cls_attn = cv2.resize(cls_attn, original_img.size, interpolation=cv2.INTER_CUBIC)
    
    return original_img, cls_attn

# ================= MAIN =================
if __name__ == "__main__":
    
    # 1. Trouver l'image
    dataset = RogerViolletDataset(DATASET_PATH, use_captions=False)
    _, target_path = dataset.get_index_by_name(TARGET_IMAGE_NAME)
    
    if not target_path:
        print(f"Image introuvable : {TARGET_IMAGE_NAME}")
        sys.exit()
    
    print(f"Analyse de l'image : {os.path.basename(target_path)}")

    # 2. Config Modèles
    models_config = [
        ("Random", None, "random"),
        ("DinoV2", None, "dinov2"),
        ("FineTuned #1", PATH_FINETUNED_1, "dinov2"),
        ("FineTuned #2", PATH_FINETUNED_2, "dinov2"),
        ("From Scratch", PATH_SCRATCH, "scratch")
    ]

    # 3. Génération Figure
    n_models = len(models_config)
    fig = plt.figure(figsize=(24, 6))
    gs = fig.add_gridspec(1, n_models + 1, width_ratios=[1] + [1]*n_models, wspace=0.05)
    
    # Affichage Image Originale (Reference)
    ax_ref = fig.add_subplot(gs[0, 0])
    img_ref = Image.open(target_path).convert('RGB')
    ax_ref.imshow(img_ref)
    ax_ref.set_title("ORIGINALE", fontsize=14, fontweight='bold', pad=10)
    ax_ref.axis('off')
    
    # Boucle sur les modèles
    for i, (name, path, mtype) in enumerate(models_config):
        print(f"-> Traitement : {name}...")
        
        # Chargement
        model, img_size = load_model_for_eval(mtype, path, DEVICE)
        
        if model:
            # Calcul Attention
            img, attn = visualize_attention(target_path, model, mtype, img_size)
            
            ax = fig.add_subplot(gs[0, i+1])
            ax.imshow(img) # Image de fond
            
            # Superposition Heatmap (Rouge/Jaune = Attention forte)
            # alpha=0.6 pour voir l'image dessous
            ax.imshow(attn, cmap='jet', alpha=0.55) 
            
            ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
            ax.axis('off')
            
            # Cadres de couleur
            color = 'gray'
            if "Random" in name: color = 'red'
            if "Official" in name: color = 'black'
            if "FineTuned" in name: color = '#2ca02c'
            if "Scratch" in name: color = 'blue'
            
            rect = plt.Rectangle((0,0), 1, 1, transform=ax.transAxes, linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Nettoyage mémoire
            del model
            torch.cuda.empty_cache()

    save_name = f"attention_maps_{TARGET_IMAGE_NAME}.png"
    plt.savefig(save_name, bbox_inches='tight', dpi=100)
    print(f"\n✅ Terminé ! Ouvrez : {save_name}")
    plt.show()