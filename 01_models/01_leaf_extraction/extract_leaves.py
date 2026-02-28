"""
Leaf Extraction Pipeline (YOLOv8 + SAM)
=======================================

This script extracts individual leaves from field images using a two-step process:
1. Object Detection (YOLOv8): Detects the bounding box of each leaf in the image.
2. Instance Segmentation (SAM): Refines the bounding box into a precise pixel-perfect mask.

The output is a set of individual, transparent background images (or black background) 
for each detected leaf, ready for downstream severity analysis.

Requirements:
    - ultralytics (YOLO)
    - segment-anything (SAM)
    - torch
    - opencv-python
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ================= Configuration =================
# Input directory containing raw field images
INPUT_DIR = "./data/raw_images"

# Output directory for extracted leaf images
OUTPUT_DIR = "./data/extracted_leaves"

# YOLO Model Path (Trained to detect 'coffee leaf' boxes)
YOLO_MODEL_PATH = "./checkpoints/yolov8_best.pt"

# SAM Model Path (Foundation model for segmentation)
# Options: 'vit_h' (huge), 'vit_l' (large), 'vit_b' (base)
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_PATH = "./checkpoints/sam_vit_h_4b8939.pth"

# =================================================

def extract_leaves():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validating paths
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        return
    if not os.path.exists(SAM_CHECKPOINT_PATH):
        print(f"Error: SAM checkpoint not found at {SAM_CHECKPOINT_PATH}")
        return

    # --- 1. Load Models ---
    print("Loading YOLOv8 model...")
    detector = YOLO(YOLO_MODEL_PATH)

    print(f"Loading SAM ({SAM_MODEL_TYPE}) model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)

    print(f"Processing images from: {INPUT_DIR}")
    
    # --- 2. Processing Loop ---
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
            continue
        
        image_path = os.path.join(INPUT_DIR, fname)
        print(f"Processing: {fname}...")

        # A. Detect Boxes with YOLO
        results = detector(image_path, verbose=False)
        # results[0].boxes.xyxy returns [x1, y1, x2, y2] for each detection
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            print(f"  No leaves detected in {fname}")
            continue

        # B. Prepare Image for SAM
        # SAM expects RGB images (0-255)
        original_image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # C. Segment and Crop Each Leaf
        for i, box in enumerate(boxes):
            # Predict mask using the box as a prompt
            # box format for SAM is same as YOLO: [x1, y1, x2, y2]
            masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)
            mask = masks[0]  # Take the best mask

            # Coordinates for cropping
            x1, y1, x2, y2 = box.astype(int)
            
            # Clamp coordinates to image boundaries to avoid errors
            h, w = original_image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            # Crop the Region of Interest (ROI)
            roi_image = original_image[y1:y2, x1:x2]
            roi_mask = mask[y1:y2, x1:x2]

            # Apply the mask: Keep leaf pixels, set background to Black (0)
            # Make sure roi_image and roi_mask have matching dimensions
            if roi_image.shape[:2] != roi_mask.shape[:2]:
                print(f"  Warning: Shape mismatch for Object {i}, skipping.")
                continue

            cutout = np.zeros_like(roi_image)
            cutout[roi_mask] = roi_image[roi_mask]

            # Save the result
            # Naming convention: {original_name}_obj{index}.png
            base_name = os.path.splitext(fname)[0]
            out_filename = f"{base_name}_obj{i}.png"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            
            # Using PNG to avoid compression artifacts, though Black background remains
            cv2.imwrite(out_path, cutout)
            
    print("✅ Extraction complete.")
    print(f"Extracted leaves saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_leaves()
