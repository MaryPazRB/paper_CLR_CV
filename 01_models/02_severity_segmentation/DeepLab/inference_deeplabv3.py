"""
DeepLabV3+ Inference Script for Coffee Leaf Rust
================================================

This script runs inference using a trained DeepLabV3+ model.
It generates BINARY masks for diseased areas:
    - 0   : Background / Healthy
    - 255 : Lesion (Rust)

Requirements:
    - torch
    - albumentations
    - segmentation-models-pytorch
    - opencv-python
"""

import os
import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ================= Configuration =================
# Path to the trained model checkpoint
MODEL_PATH = "./checkpoints/deeplab_binary_best.pth"

# Input Directory (Extracted leaves)
INPUT_DIR  = "./data/extracted_leaves"

# Output Directory (Where masks will be saved)
OUTPUT_DIR = "./data/inference_deeplab"

# Threshold to binarize probabilities (usually 0.5)
THRESHOLD = 0.5
IMG_SIZE = 512
# =================================================

def run_inference():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        return

    print("Loading DeepLabV3+ model...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,  # Weights are loaded from checkpoint
        in_channels=3,
        classes=1
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval()

    # --- Preprocessing ---
    # Must match the transforms used during training
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=mean, std=std),
    ], is_check_shapes=False)

    print(f"Processing images from: {INPUT_DIR}")
    
    processed_count = 0
    
    # --- Inference Loop ---
    # Iterate through images
    img_files = sorted(os.listdir(INPUT_DIR))
    for fname in tqdm(img_files, desc="Running Inference"):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue

        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping unreadable file: {fname}")
            continue

        original_h, original_w = img.shape[:2]

        # Convert BGR -> RGB and Apply Transforms
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img_rgb)
        input_tensor = augmented["image"]
        
        # (H,W,C) -> (1,C,H,W)
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            
            # Binarize: > 0.5 becomes 1.0, else 0.0
            mask = (probs > THRESHOLD).float().cpu().numpy()[0, 0]

        # Convert to 0 -> 255 for Binary Image
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Resize back to original image size -> Nearest Neighbor to keep it binary
        final_mask = cv2.resize(mask_uint8, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # Save
        # Depending on convention, you might want to prepend "mask_" or keep same name
        out_name = os.path.splitext(fname)[0] + ".png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), final_mask)
        processed_count += 1

    print(f"✅ Inference complete. Processed {processed_count} images.")
    print(f"Binary masks saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
