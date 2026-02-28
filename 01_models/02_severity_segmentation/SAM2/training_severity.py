"""
SAM2 Training Script for Coffee Severity Segmentation
=====================================================

This script fine-tunes the Segment Anything Model 2 (SAM2) for semantic segmentation 
of coffee leaf rust severity. It loads a dataset of images and masks, and trains 
the mask decoder of SAM2.

Methodology:
    1. Load dataset (images + binary masks).
    2. Resize and preprocess images.
    3. Generate point prompts from the ground truth masks (simulating user clicks or random points).
    4. Train the SAM2 mask decoder using a combination of IoU loss and Cross Entropy loss.
    5. Evaluate semantic segmentation performance on a test set.

Requirements:
    - torch
    - opencv-python (cv2)
    - pandas
    - scikit-learn
    - sam2 (https://github.com/facebookresearch/sam2)
"""

import os
import random
import logging
import argparse
import numpy as np
import torch
import cv2
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

# --- Import SAM2 ---
# Ensure 'sam2' is installed or in your PYTHONPATH
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: 'sam2' module not found. Please install SAM2 or check your python path.")
    exit(1)

# ================= Configuration =================
# Default paths (Can be overridden by command line arguments)
DEFAULT_DATA_DIR = "./data/coffee_severity"
DEFAULT_CFG_PATH = "./configs/sam2.1_hiera_t.yaml"
DEFAULT_CKPT_PATH = "./checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_LOG_FILE = "training.log"

# Hyperparameters
SEED = 42
TARGET_SIZE = 1024
# =================================================

def setup_logging(log_path: str):
    """Sets up logging to console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def load_dataset(data_dir: str, test_ratio=0.2):
    """
    Loads dataset from a directory containing 'images', 'masks', and a 'train.csv'.
    """
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")
    csv_path = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=SEED)

    def to_list(df_part):
        lst = []
        for _, row in df_part.iterrows():
            img_path = os.path.join(images_dir, row["imageid"])
            mask_path = os.path.join(masks_dir, row["maskid"])
            if os.path.exists(img_path) and os.path.exists(mask_path):
                lst.append({"image": img_path, "annotation": mask_path})
            else:
                logging.warning(f"Missing file: {img_path} or {mask_path}")
        return lst

    return to_list(train_df), to_list(test_df)

def read_batch(data_list, target_size=TARGET_SIZE):
    """
    Reads a random image and mask from the data list and preprocesses it.
    """
    if not data_list:
        return None, None, None, 0
    
    ent = random.choice(data_list)
    img = cv2.imread(ent["image"])
    ann = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    
    if img is None or ann is None:
        return None, None, None, 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB

    # Resize while maintaining aspect ratio (simplified)
    r = min(target_size / img.shape[1], target_size / img.shape[0])
    w_new, h_new = int(img.shape[1] * r), int(img.shape[0] * r)
    img = cv2.resize(img, (w_new, h_new))
    ann = cv2.resize(ann, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    # 1. Binarize Mask (>0 covers all non-background classes)
    binary_mask = (ann > 0).astype(np.uint8)
    
    # 2. Generate Point Prompts (Sampling from inside the mask)
    # Erode to ensure points are not on the edge
    eroded = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded > 0)
    
    points = []
    if coords.shape[0] > 0:
        # Sample up to 5 random points
        num_samples = min(5, coords.shape[0])
        idxs = np.random.choice(coords.shape[0], size=num_samples, replace=False)
        for idx in idxs:
            y, x = coords[idx]
            points.append([x, y])
    else:
        # Fallback if erosion removes everything (very small masks)
        coords = np.argwhere(binary_mask > 0)
        if coords.shape[0] > 0:
             y, x = coords[0] # Pick first point
             points.append([x, y])
    
    points = np.array(points)
    
    # Format for SAM2
    binary_mask = np.expand_dims(binary_mask, axis=0)  # shape (1, H, W)
    points = points.reshape((-1, 1, 2)) if len(points) > 0 else np.zeros((0, 1, 2))
    
    return img, binary_mask, points, 1

def build_model(cfg_path: str, checkpoint_path: str, device: str = "cuda"):
    """Loads SAM2 model and freezes the image encoder."""
    print(f"Loading SAM2 from {checkpoint_path}...")
    model = build_sam2(cfg_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    
    # Freeze image encoder to save memory and time
    for param in predictor.model.image_encoder.parameters():
        param.requires_grad = False
    
    # Train only the mask decoder and prompt encoder
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    
    return predictor

def train_one_step(predictor, optimizer, scheduler, scaler,
                   train_data, step, accumulation_steps, logger, mean_iou):
    """Performs one training step."""
    predictor.model.train()
    
    with autocast():
        img, mask_np, points, num_masks = read_batch(train_data)
        
        # Skip if bad data or no points generated
        if img is None or num_masks == 0 or points.shape[0] == 0:
            return mean_iou

        predictor.set_image(img)
        
        # Prepare inputs
        input_label = np.ones((points.shape[0], 1)) # All points are foreground (1)
        mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
            points, input_label, box=None, mask_logits=None, normalize_coords=True
        )

        # Forward Pass
        sparse_emb, dense_emb = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )
        
        batched = unnorm_coords.shape[0] > 1
        high_feats = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
        
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
            repeat_image=batched,
            high_res_features=high_feats
        )
        
        # Post-process and Calculate Loss
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        gt = torch.tensor(mask_np.astype(np.float32)).cuda()
        prd = torch.sigmoid(prd_masks[:, 0]) # Take first mask
        
        # Binary Cross Entropy + IoU Loss
        seg_loss = (-gt * torch.log(prd + 1e-6) - (1 - gt) * torch.log(1 - prd + 1e-6)).mean()
        
        inter = (gt * (prd > 0.5)).sum()
        union = gt.sum() + (prd > 0.5).sum() - inter
        iou = (inter / (union + 1e-6)).item()
        
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + 0.05 * score_loss
        loss = loss / accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
        
        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Moving average IoU
        mean_iou = mean_iou * 0.99 + 0.01 * iou
        
        if step % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Step {step}: LR={lr:.6e}, IoU={mean_iou:.4f}, Loss={seg_loss.item():.4f}")
            
    return mean_iou

def run_training(args):
    logger = setup_logging(args.log_file)
    logger.info(f"Initializing training on device: {args.device}")
    set_seeds(args.seed)

    # Load Data
    try:
        train_data, test_data = load_dataset(args.data_dir, test_ratio=args.test_ratio)
        logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Build Model
    predictor = build_model(args.cfg_path, args.ckpt_path, device=args.device)
    
    # Optimizer
    optimizer = AdamW(predictor.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = GradScaler()

    mean_iou = 0.0
    logger.info("Starting training loop...")
    
    for step in range(1, args.max_steps + 1):
        mean_iou = train_one_step(predictor, optimizer, scheduler, scaler,
                                  train_data, step, args.accum_steps, logger, mean_iou)
        
        # Basic validation (could be improved to run on full test set)
        if step % 500 == 0:
             # Save checkpoint
            fname = f"checkpoint_step_{step}.pt"
            torch.save(predictor.model.state_dict(), fname)
            logger.info(f"Saved checkpoint: {fname}")

    logger.info("Training finished.")

def parse_args():
    p = argparse.ArgumentParser(description="Train SAM2 for Coffee Leaf Rust Severity")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to dataset folder")
    p.add_argument("--cfg-path", type=str, default=DEFAULT_CFG_PATH, help="Path to SAM2 config YAML")
    p.add_argument("--ckpt-path", type=str, default=DEFAULT_CKPT_PATH, help="Path to pretrained SAM2 checkpoint")
    p.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    p.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE, help="Path to log file")
    p.add_argument("--max-steps", type=int, default=6000, help="Total training steps")
    p.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--step-size", type=int, default=2000, help="Scheduler step size")
    p.add_argument("--gamma", type=float, default=0.6, help="Scheduler gamma")
    p.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of test set")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
