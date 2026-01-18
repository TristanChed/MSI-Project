import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from utils import (load_model_for_eval, get_transforms, RogerViolletDataset, PATH_FINETUNED_1, PATH_FINETUNED_2, PATH_SCRATCH, DATASET_PATH, DEVICE)

BATCH_SIZE = 64
TOP_K = 5
TARGET_IMAGE_NAME = "467" 

def find_neighbors_optimized(dataset, target_idx, model_path, model_type, name):
    print(f"\n--- Recherche avec : {name} ---")
    
    model, img_size = load_model_for_eval(model_type, model_path, DEVICE)
    if model is None: return None, None
            
    dataset.transform = get_transforms(img_size)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    embeddings = []
    
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(loader, desc="Extraction"):
            imgs = imgs.to(DEVICE)
            if model_type in ["dinov2", "random"]:
                feats = model.forward_features(imgs)["x_norm_clstoken"]
            else:
                feats = model(imgs)
            embeddings.append(feats.cpu())
            
    all_embs = torch.cat(embeddings, dim=0)
    all_embs = F.normalize(all_embs, p=2, dim=1)
    
    query_vec = all_embs[target_idx].unsqueeze(0)
    scores = torch.mm(query_vec, all_embs.t()).squeeze(0)
    top_scores, top_indices = torch.topk(scores, k=TOP_K+1)
    
    top_indices = top_indices.numpy()
    top_scores = top_scores.numpy()
    
    result_paths = []
    result_scores = []
    
    for i in range(len(top_indices)):
        idx = top_indices[i]
        if idx == target_idx: continue
        full_path = dataset.samples[idx][0]
        result_paths.append(full_path)
        result_scores.append(top_scores[i])
        if len(result_paths) == TOP_K: break
        
    dataset.transform = None
    del model, all_embs
    torch.cuda.empty_cache()
    return result_paths, result_scores

if __name__ == "__main__":
    
    dataset = RogerViolletDataset(DATASET_PATH, use_captions=False)
    target_idx, target_path = dataset.get_index_by_name(TARGET_IMAGE_NAME)
    
    if target_idx is None:
        print(f"ERREUR : Image '{TARGET_IMAGE_NAME}' introuvable.")
        sys.exit(1)
    print(f"Cible : {os.path.basename(target_path)}")
    
    models_config = [
        ("Random", None, "random"),
        ("DinoV2", None, "dinov2"),
        ("FineTuned #1", PATH_FINETUNED_1, "dinov2"),
        ("FineTuned #2", PATH_FINETUNED_2, "dinov2"),
        ("From Scratch", PATH_SCRATCH, "scratch")
    ]
    
    results = {}
    for name, path, mtype in models_config:
        paths, scores = find_neighbors_optimized(dataset, target_idx, path, mtype, name)
        results[name] = (paths, scores)

    print("\nGénération de l'image comparative optimisée...")
    n_models = len(models_config)
    
    # --- OPTIMISATION TAILLE ---
    # On élargit la figure (width=24) et on réduit la hauteur par ligne (3.0) 
    # car on supprime les titres au-dessus des images
    fig = plt.figure(figsize=(24, 3.0 * n_models))
    
    # WSPACE=0 et HSPACE=0 pour coller les images
    gs = fig.add_gridspec(n_models, TOP_K + 1, 
                          width_ratios=[1.3] + [1]*TOP_K, 
                          wspace=0.01, # Espace minime
                          hspace=0.01) # Espace minime
    
    # --- IMAGE REQUÊTE ---
    ax_query = fig.add_subplot(gs[:, 0])
    ax_query.imshow(Image.open(target_path).convert('RGB'))
    
    # Titre incrusté en bas de l'image requête pour gagner de la place en haut
    ax_query.text(0.5, 0.02, "REQUÊTE\n" + os.path.basename(target_path),
                  transform=ax_query.transAxes, ha='center', va='bottom',
                  color='white', fontweight='bold', fontsize=12,
                  bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    
    ax_query.axis('off')
    # Cadre autour de la requête
    rect_q = plt.Rectangle((0,0), 1, 1, transform=ax_query.transAxes, linewidth=4, edgecolor='black', facecolor='none')
    ax_query.add_patch(rect_q)
    
    # --- RÉSULTATS ---
    for row_idx, (name, _, _) in enumerate(models_config):
        if name not in results or results[name][0] is None: continue
        r_paths, r_scores = results[name]
        
        for k in range(TOP_K):
            ax = fig.add_subplot(gs[row_idx, k+1])
            try: ax.imshow(Image.open(r_paths[k]).convert('RGB'))
            except: pass
            
            # --- INCUSTATION DU SCORE ---
            # Au lieu d'un titre, on met le texte DANS l'image
            ax.text(0.03, 0.97, f"{r_scores[k]:.3f}", 
                    transform=ax.transAxes, ha='left', va='top',
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
            
            # --- NOM DU MODÈLE ---
            # On l'affiche verticalement à gauche de la première image de résultat
            # ou horizontalement très serré
            if k == 0:
                ax.text(-0.05, 0.5, name, transform=ax.transAxes, 
                        va='center', ha='right', fontsize=13, fontweight='bold', rotation=0)
            
            ax.axis('off')


    # Sauvegarde SANS AUCUNE MARGE BLANCHE (pad_inches=0)
    save_name = f"comparaison_MAX_{TARGET_IMAGE_NAME}.png"
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()