"""
DeepLabV3+ Training Script for Coffee Leaf Rust
===============================================

This script trains a DeepLabV3+ model (ResNet50 encoder) for binary semantic segmentation:
    - Class 0: Background / Healthy
    - Class 1: Lesion (Rust)

Requirements:
    - torch
    - albumentations
    - segmentation-models-pytorch (smp)
    - opencv-python
"""

import os
import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= Configuration =================
# Dataset Paths
TRAIN_IMG_DIR = "./data/dataset/images/train"
TRAIN_MASK_DIR = "./data/dataset/masks/train"

VAL_IMG_DIR   = "./data/dataset/images/valid"
VAL_MASK_DIR  = "./data/dataset/masks/valid"

# Output Paths
SAVE_LAST = "./checkpoints/deeplab_binary_last.pth"
SAVE_BEST = "./checkpoints/deeplab_binary_best.pth"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 25
IMG_SIZE = 512
LEARNING_RATE = 1e-4
# =================================================

class LeafDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Assumption: mask has same basename + _mask.png
        # Adjust this logic if your naming convention differs
        base = os.path.splitext(img_name)[0]
        mask_name = base + "_mask.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0) # Read as grayscale
        if mask is None:
             # Fallback: try reading with original name if _mask suffix not used
            mask_name = img_name
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise RuntimeError(f"Cannot read mask for: {img_name}")

        # Binary mask: lesion = 1, background = 0
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # To Tensor: (H, W, C) -> (C, H, W)
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask

def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
    ], is_check_shapes=False)

    val_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=mean, std=std),
    ], is_check_shapes=False)
    
    return train_tf, val_tf

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running += loss.item()
        pbar.set_postfix(loss=loss.item())

    return running / len(loader)

def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            running += loss.item()
            pbar.set_postfix(loss=loss.item())

    return running / len(loader)

def main():
    print("\n=== Binary DeepLabV3+ Training ===")
    
    # Check Directories
    if not os.path.exists(TRAIN_IMG_DIR):
        print(f"Error: Training directory not found: {TRAIN_IMG_DIR}")
        return
    
    os.makedirs(os.path.dirname(SAVE_BEST), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Transforms & Datasets
    train_tf, val_tf = get_transforms()
    train_ds = LeafDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_tf)
    val_ds   = LeafDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_tf)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    # Loss & Optimizer
    # Combine Dice Loss and BCE for robust segmentation training
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = torch.nn.BCEWithLogitsLoss()
    
    def criterion(logits, targets):
        return dice_loss(logits, targets) + bce_loss(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss   = val_epoch(model, val_loader, criterion, device, epoch)

        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"✔ Best model saved to: {SAVE_BEST}")
    
    torch.save(model.state_dict(), SAVE_LAST)
    print("Training finished.")

if __name__ == "__main__":
    main()
