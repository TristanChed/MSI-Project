import torch
import torch.nn as nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import utils
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import (DATASET_PATH, DEVICE)

BATCH_SIZE = 64
EPOCHS = 10

# LR différents pour head et bacnckbone 
LR_HEAD = 0.00001       
LR_BACKBONE = 0.000001 
MIN_LR_RATIO = 0.1      
CHECKPOINT_DIR = "checkpoints_2"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Fine-tuning DINOv2 (Differential LR + Partial Freeze) sur : {DEVICE}")

transform = DINOTransform(
    global_crop_size=224,    
    local_crop_size=98,
    n_local_views=8,
)

dataset = utils.RecursiveImageDataset(root_dir=DATASET_PATH, transform=transform)

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True 
)

print("Chargement des poids DinoV2")
official_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

class DINOv2Wrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        result = self.backbone.forward_features(x)
        return result["x_norm_clstoken"]

backbone_student = DINOv2Wrapper(official_backbone).to(DEVICE)
backbone_teacher = DINOv2Wrapper(torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')).to(DEVICE)

class DINOModel(nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.backbone = backbone
        self.head = DINOProjectionHead(
            input_dim, 
            hidden_dim=2048, 
            bottleneck_dim=256, 
            output_dim=65536,
            freeze_last_layer=1 
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

student = DINOModel(backbone_student, input_dim=384).to(DEVICE)
teacher = DINOModel(backbone_teacher, input_dim=384).to(DEVICE)

teacher.load_state_dict(student.state_dict())
deactivate_requires_grad(teacher)

# On gèle le backbone
for param in student.backbone.parameters():
    param.requires_grad = False

# Fine Tuning 1 : on débloque seulement la dernière couche de norm et le block 11
# Fine tuning 2 : on débloque les block 9, 10, 11 et la couche de Norm
blocks_to_unfreeze = ["blocks.9", "blocks.10", "blocks.11", "norm"]
for name, param in student.backbone.backbone.named_parameters():
    if any(x in name for x in blocks_to_unfreeze):
        param.requires_grad = True


# Backbone
backbone_params = [p for p in student.backbone.parameters() if p.requires_grad]

# Head
head_params = [p for p in student.head.parameters()]

# Création des groupes pour l'optimiseur
param_groups = [
    {'params': backbone_params, 'lr': LR_BACKBONE, 'name': 'backbone'},
    {'params': head_params, 'lr': LR_HEAD, 'name': 'head'}
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.04)

# Permet de réduire les deux LR proportionnellemnt
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS, 
    eta_min=0
)

scaler = torch.amp.GradScaler('cuda')
criterion = DINOLoss(output_dim=65536, warmup_teacher_temp_epochs=1).to(DEVICE)

n_backbone = sum(p.numel() for p in backbone_params)
n_head = sum(p.numel() for p in head_params)
print(f"--> Config Optimizer : Backbone LR={LR_BACKBONE} ({n_backbone:,} params) | Head LR={LR_HEAD} ({n_head:,} params)")

# Utilisation de tensorboard pour sauvgarder les logs
run_name = "runs/finetune_2"
writer = SummaryWriter(log_dir=run_name)
print(f"Logs : {run_name}")

# TRAINING LOOP -
print("Starting training")
global_step = 0

for epoch in range(EPOCHS):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, EPOCHS, 0.996, 1)
    
    # Récupération des LRs pour affichage (Index 0 = Backbone, Index 1 = Head)
    lr_back = optimizer.param_groups[0]['lr']
    lr_head = optimizer.param_groups[1]['lr']
    
    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}] | LR_Back: {lr_back:.1e} | LR_Head: {lr_head:.1e}")
    
    for views, _, _ in progress_bar:
        views = [view.to(DEVICE, non_blocking=True) for view in views]
        global_views = views[:2]

        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                teacher_out = [teacher(view) for view in global_views]
            student_out = [student(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Clipping global
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        update_momentum(student, teacher, m=momentum_val)
        
        # LOGGING
        loss_val = loss.item()
        total_loss += loss_val
        
        writer.add_scalar("Train/Batch_Loss", loss_val, global_step)
        writer.add_scalar("LR/Backbone", lr_back, global_step)
        writer.add_scalar("LR/Head", lr_head, global_step)
        global_step += 1
        
        progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})

    scheduler.step()

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
    print(f"Fin Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

    # Sauvegarde
    save_name = f"dino_diff_lr_ep{epoch+1:02d}_loss{avg_loss:.4f}.pth"
    torch.save(student.backbone.backbone.state_dict(), os.path.join(CHECKPOINT_DIR, save_name))

torch.save(student.backbone.backbone.state_dict(), os.path.join(CHECKPOINT_DIR, "dino_fine_tunned_1.pth"))
writer.close()
print("Terminé.")