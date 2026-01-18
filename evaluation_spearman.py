import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import gc
from collections import Counter
from utils import (load_model_for_eval, get_transforms, RogerViolletDataset,PATH_FINETUNED_1, PATH_FINETUNED_2, PATH_SCRATCH, DATASET_PATH, DEVICE)

MAX_SAMPLES = 2500 
BATCH_SIZE = 64

def extract_features_optimized(dataset, model_path, model_type, name):
    print(f"Évaluation : {name} ---")
    
    model, img_size = load_model_for_eval(model_type, model_path, DEVICE)
    if model is None: return None, None

    dataset.transform = get_transforms(img_size)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    embeddings = []
    all_series_ids = []
    
    with torch.no_grad():
        for imgs, _, series_ids, _ in tqdm(loader, desc=f"Extract {name}"):
            imgs = imgs.to(DEVICE)
            if model_type in ["dinov2", "random"]:
                feats = model.forward_features(imgs)["x_norm_clstoken"]
            else:
                feats = model(imgs) 
            embeddings.append(feats.cpu().numpy())
            all_series_ids.extend(series_ids)
            
    dataset.transform = None # Clean up
    del model
    torch.cuda.empty_cache()
    
    return np.concatenate(embeddings), all_series_ids

# Spearman
def calculate_spearman(img_embeddings, txt_embeddings):
    img_emb = img_embeddings / np.linalg.norm(img_embeddings, axis=1, keepdims=True)
    txt_emb = txt_embeddings / np.linalg.norm(txt_embeddings, axis=1, keepdims=True)
    sim_img = np.dot(img_emb, img_emb.T)
    sim_txt = np.dot(txt_emb, txt_emb.T)
    inds = np.triu_indices(sim_img.shape[0], k=1)
    return spearmanr(sim_img[inds], sim_txt[inds])[0]

def calculate_recall_at_k(img_embeddings, series_ids, ks=[1, 10]):
    embeddings = torch.from_numpy(img_embeddings).to(DEVICE)
    import torch.nn.functional as F
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(embeddings, embeddings.t())
    sim_matrix.fill_diagonal_(-1)
    n_samples = embeddings.size(0)
    _, topk_indices = sim_matrix.topk(max(ks), dim=1)
    topk_indices = topk_indices.cpu().numpy()
    recalls = {k: 0 for k in ks}
    valid_queries = 0
    series_counts = Counter(series_ids)
    for i in range(n_samples):
        true_id = series_ids[i]
        if series_counts[true_id] < 2: continue
        valid_queries += 1
        retrieved = [series_ids[idx] for idx in topk_indices[i]]
        for k in ks:
            if true_id in retrieved[:k]: recalls[k] += 1
    if valid_queries == 0: return {k: 0.0 for k in ks}
    return {k: (recalls[k] / valid_queries) * 100 for k in ks}

if __name__ == "__main__":
    
    # On récuềre les images avec les légendes
    full_ds = RogerViolletDataset(DATASET_PATH, use_captions=True)
    total = len(full_ds)
    
    if total > MAX_SAMPLES:
        indices = torch.randperm(total)[:MAX_SAMPLES].tolist()
        dataset = RogerViolletDataset(DATASET_PATH, use_captions=True, fixed_indices=indices)
    else:
        dataset = full_ds
        
    # On récupère les embessings de texte via un Sentence Transformer
    print("\n[1/6] Embeddings Texte...")
    text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    txt_embs = []

    def collate_txt(batch): return [item[1] for item in batch]
    txt_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_txt)
    
    for texts in tqdm(txt_loader):
        txt_embs.append(text_model.encode(texts))
    text_embeddings = np.concatenate(txt_embs)
    del text_model
    gc.collect()
    
    #ÉVALUATION
    results_table = []
    models_to_test = [
        ("Random", None, "random"),
        ("DinoV2", None, "dinov2"),
        ("FineTuned #1", PATH_FINETUNED_1, "dinov2"),
        ("FineTuned #2", PATH_FINETUNED_2, "dinov2"),
        ("Model from scracth", PATH_SCRATCH, "scratch")
    ]
    
    for name, path, mtype in models_to_test:
        feats, s_ids = extract_features_optimized(dataset, path, mtype, name)
        if feats is not None:
            spearman = calculate_spearman(feats, text_embeddings)
            recalls = calculate_recall_at_k(feats, s_ids, ks=[1, 10])
            results_table.append({
                "Modèle": name, "Spearman": spearman, "R@1": recalls[1], "R@10": recalls[10]
            })
            
    print("\n" + "="*65)
    print(f"{'Modèle':<18} | {'Spearman':<10} | {'R@1':<10} | {'R@10':<10}")
    print("-" * 65)
    for res in results_table:
        print(f"{res['Modèle']:<18} | {res['Spearman']:.4f}     | {res['R@1']:.2f}%      | {res['R@10']:.2f}%")
    print("="*65)