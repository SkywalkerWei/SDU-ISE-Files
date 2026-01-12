import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import yaml
from tqdm import tqdm
import os

from models import Model
from dataset import COCODataset
from loss import ComputeLoss

def train(cfg, device):
    # Create directory for weights
    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Load dataset info
    with open(cfg['data'], 'r') as f:
        data_dict = yaml.safe_load(f)
    
    nc = int(data_dict['nc'])
    
    # Model
    model = Model(nc=nc).to(device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=cfg['lr0'], betas=(0.9, 0.999))
    
    # Scheduler
    scheduler = StepLR(optimizer, step_size=cfg['scheduler_step_size'], gamma=0.1)
    
    # Loss
    compute_loss = ComputeLoss(model)

    # Dataset
    train_dataset = COCODataset(
        annotation_file=os.path.join(data_dict['train'], '../annotations/instances_train2017.json'),
        img_dir=data_dict['train'],
        img_size=cfg['img_size'],
        is_train=True,
        debug=cfg['debug']
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=COCODataset.collate_fn
    )

    print(f"Starting training for {cfg['epochs']} epochs...")
    for epoch in range(cfg['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        total_loss = 0
        
        for i, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device)
            
            # Forward
            preds = model(imgs)
            
            # Loss
            loss, loss_items = compute_loss(preds, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': total_loss / (i + 1),
                'box_loss': loss_items[0].item(),
                'obj_loss': loss_items[1].item(),
                'cls_loss': loss_items[2].item()
            })
            
        scheduler.step()
        
        # Save model
        torch.save(model.state_dict(), os.path.join(weights_dir, f'epoch_{epoch+1}.pt'))
        print(f"Model saved to {os.path.join(weights_dir, f'epoch_{epoch+1}.pt')}")


if __name__ == '__main__':
    config = {
        'data': 'voc.yaml',
        'epochs': 50,
        'batch_size': 16,
        'lr0': 0.001,
        'img_size': 640,
        'scheduler_step_size': 15,
        'debug': True # Set to False to train on the full dataset
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(config, device)