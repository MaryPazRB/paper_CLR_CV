"""
Multi-class Inference for Coffee Leaf Rust
==========================================

This script performs semantic segmentation on leaf images using a fine-tuned SAM (Segment Anything Model) adapter.
It segments the image into three classes:
    0: Background
    1: Diseased Area (Rust)
    2: Healthy Leaf Area

The output includes:
    - Overlay images (original image + colored segmentation mask)
    - Raw mask files (optional)

Requirements:
    - torch
    - segment-anything
    - numpy
    - cv2
    - PIL
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

# ================= Configuration =================
# Model Checkpoints
# MODEL_TYPE: The size of the SAM model used (e.g., 'vit_b', 'vit_l', 'vit_h')
MODEL_TYPE = "vit_b"

# Path to the base SAM checkpoint (download from Meta AI)
CHECKPOINT_PATH = "./checkpoints/sam_vit_b_01ec64.pth"

# Path to your trained Segmentation Head (adapter)
SEG_HEAD_PATH = "./checkpoints/seg_head_multiclass_epoch10.pth"

# Input/Output Directories
IMAGE_FOLDER = "./data/inference_images"
OUTPUT_FOLDER = "./data/inference_output"

# Visualization Colors (B, G, R) - OpenCV uses BGR
# Class 1: Disease (Red), Class 2: Leaf (Green)
COLOR_MAP = {
    1: (0, 0, 255),  # Red
    2: (0, 255, 0)   # Green
}
# =================================================

class SAMSegHead(nn.Module):
    """
    Segmentation Head for SAM.
    Adapts SAM's image embeddings to multi-class segmentation logits.
    """
    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)

def run_inference():
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading SAM model...")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device)
    except FileNotFoundError:
        print(f"Error: SAM checkpoint not found at {CHECKPOINT_PATH}")
        return

    print("Loading Segmentation Head...")
    seg_head = SAMSegHead().to(device)
    try:
        seg_head.load_state_dict(torch.load(SEG_HEAD_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Segmentation head checkpoint not found at {SEG_HEAD_PATH}")
        return

    seg_head.eval()
    sam.eval()

    # Preprocessing transform
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor()
    ])

    print(f"Starting inference on images in {IMAGE_FOLDER}...")

    # --- Inference Loop ---
    processed_count = 0
    for img_name in os.listdir(IMAGE_FOLDER):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get SAM embeddings
            feats = sam.image_encoder(img_tensor)
            # Predict masks
            logits = seg_head(feats)
            # Resize back to original image size
            logits = F.interpolate(logits, size=original_size[::-1], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # --- Visualization ---
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        overlay = image_bgr.copy()

        for cls_id, color in COLOR_MAP.items():
            mask = preds == cls_id
            overlay[mask] = color

        # Blend original image with overlay (Opacity: 0.3)
        blended = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)
        
        # Save Result
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(output_path, blended)

        # Optional: Save raw mask (scaled for visibility if needed, or raw values)
        # cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"mask_{img_name}"), preds * 100)
        
        processed_count += 1
        print(f"Processed: {img_name}")

    print(f"✅ Multi-class inference complete! Processed {processed_count} images.")
    print(f"Results saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_inference()
