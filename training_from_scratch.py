import torch
from torch import nn
import lightly.data as data
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torchvision.models.vision_transformer import VisionTransformer
import utils
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DinoVisionTransformer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import (DATASET_PATH, DEVICE)


BATCH_SIZE = 64             
IMAGE_SIZE = 256            
PATCH_SIZE = 16              
EPOCHS = 50
print(f"Entraînement lancé sur : {DEVICE}")

transform = DINOTransform(
    global_crop_size=256,    
    local_crop_size=96,      
    n_local_views=8,
)

dataset = utils.RecursiveImageDataset(root_dir=DATASET_PATH, transform=transform)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True 
)


def get_vit_small_256():
    model = DinoVisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,      
        mlp_dim=1536,
        num_classes=0        
    )
    return model

backbone = get_vit_small_256()
backbone.heads = nn.Identity()

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

student = DINOModel(backbone, input_dim=384) 
teacher = DINOModel(get_vit_small_256(), input_dim=384)
teacher.backbone.heads = nn.Identity()

teacher.load_state_dict(student.state_dict())
deactivate_requires_grad(teacher)

# GPU
student = student.to(DEVICE)
teacher = teacher.to(DEVICE)

criterion = DINOLoss(output_dim=65536, warmup_teacher_temp_epochs=5).to(DEVICE)
optimizer = torch.optim.AdamW(student.parameters(), lr=0.0001, weight_decay=0.04)

# Scaler pour le Mixed Precision
scaler = torch.amp.GradScaler('cuda') 

print(f"Démarrage sur {len(dataset)} images...")

log_dir = "runs/dino_experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

print(f"TensorBoard : Logs enregistrés dans {log_dir}")

# Suivi du nombre de pas dans le training
global_step = 0 

# Training
epoch_losses = []

for epoch in range(EPOCHS):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, EPOCHS, 0.996, 1)

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    
    for views, _, _ in progress_bar:
        # Transfert asynchrone
        views = [view.to(DEVICE, non_blocking=True) for view in views]
        global_views = views[:2]

        # Mixed Precision Context
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                teacher_out = [teacher(view) for view in global_views]
            student_out = [student(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)

        # Backward & Step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        # Update Teacher
        update_momentum(student, teacher, m=momentum_val)
        
        loss_val = loss.item()
        total_loss += loss_val
        
        writer.add_scalar("Train/Batch_Loss", loss_val, global_step)
        writer.add_scalar("Hyperparameters/Learning_Rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("Hyperparameters/Momentum", momentum_val, global_step)
        
        global_step += 1 

        current_avg = total_loss / (progress_bar.n + 1)
        progress_bar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "avg": f"{current_avg:.4f}",
            "m": f"{momentum_val:.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    
    writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
    
    print(f"Fin Epoch {epoch + 1} | Loss Moyenne: {avg_loss:.4f}")

torch.save(student.backbone.state_dict(), "dino_vitsmall_256_50.pth")
writer.close() # Fermer proprement le writer