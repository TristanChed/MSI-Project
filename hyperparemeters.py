import torch
from torch import nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torchvision.models.vision_transformer import VisionTransformer
import utils
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import DinoVisionTransformer

BATCH_SIZE = 64             
IMAGE_SIZE = 256            
PATCH_SIZE = 16              
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_TEST = 4 

LEARNING_RATES = [0.0001, 0.0005, 0.001] 

transform = DINOTransform(global_crop_size=256, local_crop_size=96, n_local_views=8)
DATASET_PATH = os.path.expanduser("/home/tristan-chedeville/Bureau/corpus_RogerViollet_uniformized")
dataset = utils.RecursiveImageDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

def get_vit_small_256():
    return DinoVisionTransformer(image_size=IMAGE_SIZE,
                                 patch_size=PATCH_SIZE,
                                 num_layers=12, num_heads=6,
                                 hidden_dim=384, mlp_dim=1536,
                                 num_classes=0)

class DINOModel(nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.backbone = backbone
        self.head = DINOProjectionHead(input_dim, hidden_dim=2048, bottleneck_dim=256, output_dim=65536, freeze_last_layer=1)
    def forward(self, x):
        return self.head(self.backbone(x))


for lr in LEARNING_RATES:
    print(f"\n{'='*40}")
    print(f"TEST LANCÉ AVEC LR : {lr} ---")
    print(f"{'='*40}")
    
    backbone_student = get_vit_small_256()
    backbone_student.heads = nn.Identity()
    student = DINOModel(backbone_student, input_dim=384).to(DEVICE)
    
    backbone_teacher = get_vit_small_256()
    backbone_teacher.heads = nn.Identity()
    teacher = DINOModel(backbone_teacher, input_dim=384).to(DEVICE)
    
    teacher.load_state_dict(student.state_dict())
    deactivate_requires_grad(teacher)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.04)
    scaler = torch.amp.GradScaler('cuda') 
    criterion = DINOLoss(output_dim=65536, warmup_teacher_temp_epochs=1).to(DEVICE) # Warmup raccourci

    run_name = f"runs/SWEEP_LR_{lr}_{datetime.datetime.now().strftime('%H%M')}"
    writer = SummaryWriter(log_dir=run_name)
    global_step = 0
    
    for epoch in range(EPOCHS_TEST):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, EPOCHS_TEST, 0.996, 1)
        
        progress_bar = tqdm(dataloader, desc=f"LR {lr} | Ep {epoch+1}/{EPOCHS_TEST}")
        
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
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            update_momentum(student, teacher, m=momentum_val)
            
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            total_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
        print(f"Fin Epoch {epoch+1} | Loss Moyenne: {avg_loss:.4f}")
        
    writer.close()
    
    del student, teacher, optimizer, scaler, criterion
    torch.cuda.empty_cache()

