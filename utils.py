import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from PIL import Image

# Models et dataset
PATH_FINETUNED_1 = "models/dino_fine_tuned_1.pth"
PATH_FINETUNED_2 = "models/dino_fine_tuned_2.pth"
PATH_SCRATCH     = "models/dino_from_scratch.pth"  
DATASET_PATH     = "/home/tristan-chedeville/Bureau/corpus_RogerViollet_uniformized"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model à entrainer
class DinoVisionTransformer(nn.Module):
    """Wrapper pour charger les poids entraînés via Torchvision/Lightly"""
    def __init__(self):
        super().__init__()
        self.backbone = models.vision_transformer.VisionTransformer(
            image_size=256, patch_size=16, num_layers=12, num_heads=6, 
            hidden_dim=384, mlp_dim=1536, num_classes=0
        )
    def forward(self, x):
        x = self.backbone._process_input(x)
        n = x.shape[0]
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)
        return x[:, 0]


def load_model_for_eval(model_type, model_path=None, device=DEVICE):
    """
    Charge n'importe quel modèle (DINOv2, Random, Scratch) et retourne le modèle + la taille d'image requise.
    """
    img_size = 252 # Taille par défaut pour DINOv2
    model = None

    print(f"-> Chargement architecture : {model_type}")

    # DinoV2
    if model_type == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if model_path:
            if os.path.exists(model_path):
                print(f"Model chargé : {os.path.basename(model_path)}")
                state = torch.load(model_path, map_location=device)
                clean = {k.replace("backbone.", "").replace("module.", ""): v for k, v in state.items() if "head" not in k}
                model.load_state_dict(clean, strict=False)
            else:
                print(f"Model introuvables : {model_path}. Utilisation de DinoV2.")

    # DinoV2 avec poids random
    elif model_type == "random":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        model.apply(init_weights)

    # Model from scratch
    elif model_type == "scratch":
        img_size = 256 
        model = DinoVisionTransformer()
        if model_path and os.path.exists(model_path):
            print(f"Model chargé : {os.path.basename(model_path)}")
            state = torch.load(model_path, map_location=device)
            clean = {}
            for k, v in state.items():
                # Nettoyage spécifique Lightly/Torchvision
                new_k = k.replace("teacher.backbone.", "").replace("student.backbone.", "").replace("backbone.", "")
                clean[new_k] = v
            model.load_state_dict(clean, strict=False)
        else:
            print(f"Model introuvable : {model_path}")
            return None, None

    model.to(device).eval()
    return model, img_size

def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Dataset

# Recherche des images dans le dossier et tous les sous dissuers
class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Sélection des formats des images
        valid_extensions = ('.jpg', '.jpeg', '.png')
        print(f"Exploration des dossiers dans : {root_dir} ...")
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {root_dir}.")
            
        print(f"Dataset chargé : {len(self.image_paths)} images trouvées.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                views = self.transform(image)
            return views, 0, img_path
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

class RogerViolletDataset(Dataset):
    def __init__(self, root_dir, use_captions=False, fixed_indices=None):
        self.samples = [] 
        self.transform = None 
        
        print(f"Indexation du dataset ({'Avec descriptions' if use_captions else 'Images seules'})...")
        
        if use_captions:
            # Besoin des légendes pour le calcul du score de spearman uniquement
            excel_files = [os.path.join(r, f) for r, d, fs in os.walk(root_dir) for f in fs if f.endswith(('.xlsx', '.xls'))]
            dfs = []
            for ef in excel_files:
                try:
                    df = pd.read_excel(ef, dtype=str)
                    df.columns = [c.lower() for c in df.columns] 
                    if 'descripteurs' in df.columns and 'photoref' in df.columns:
                        dfs.append(df[['photoref', 'descripteurs']])
                except: pass
            
            if not dfs: raise ValueError("Aucun Excel trouvé pour les légendes.")
            full_df = pd.concat(dfs, ignore_index=True)
            full_df['photoref'] = full_df['photoref'].astype(str).str.strip()
            full_df = full_df.dropna(subset=['descripteurs'])
            ref_to_caption = dict(zip(full_df['photoref'], full_df['descripteurs']))
            
            valid_exts = ('.jpg', '.jpeg', '.png')
            all_candidates = []
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        img_id = os.path.splitext(file)[0]
                        if img_id in ref_to_caption:
                            series_id = img_id.split('-')[0] if '-' in img_id else img_id
                            all_candidates.append((os.path.join(root, file), ref_to_caption[img_id], series_id))
        else:
            # Load du dataset sans légendes
            all_candidates = []
            valid_exts = ('.jpg', '.jpeg', '.png')
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        # Pour le format, on met "No Text" et l'ID série est le nom du fichier
                        all_candidates.append((os.path.join(root, file), "No Text", file))
            all_candidates.sort()

        # Filtrage si indices fixés
        if fixed_indices is not None:
            self.samples = [all_candidates[i] for i in fixed_indices]
        else:
            self.samples = all_candidates

        print(f"-> {len(self.samples)} images indexées.")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, txt, series_id = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform: img = self.transform(img)
        # On retourne aussi le chemin pour la visualisation
        return img, str(txt), str(series_id), path

    def get_index_by_name(self, name_fragment):
        """Utile pour la visualisation : trouver l'index d'une image par son nom"""
        for i, sample in enumerate(self.samples):
            # sample[0] est le chemin complet
            if name_fragment in os.path.basename(sample[0]):
                return i, sample[0]
        return None, None