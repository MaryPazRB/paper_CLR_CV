"""
SAM3 Inference Script for Lesion Segmentation (Text Prompt)
===========================================================

This script runs inference using the SAM3 model with TEXT PROMPTS.
It is designed to batch process a folder of images.

Usage:
    - Define `TEXT_PROMPT` in the Configuration (e.g., "orange spot", "lesion").
    - The model automatically segments areas matching the text description.
    - Output is a BINARY mask (0=Background, 255=Target).

Requirements:
    - torch
    - PIL (Pillow)
    - numpy
    - sam3 (custom library)
    - huggingface_hub

Prerequisites (SAM3 Specific):
    *   **Hugging Face Access**: You must have a Hugging Face account and valid access permissions for the SAM3 model.
    *   **Login**: You must be logged in locally to download/access the model weights.
        Run the following command in your terminal and paste your access token:
        `huggingface-cli login`
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

# Ensure user is logged in (optional check)
try:
    from huggingface_hub import login
except ImportError:
    pass 

# Try importing SAM3
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Error: Could not import 'sam3'. Ensure the submodule is present.")

# ================= Configuration =================
# Input Directory (Folder containing leaf images)
INPUT_DIR = "./data/images"

# Output Directory (Where masks will be saved)
OUTPUT_DIR = "./data/inference_sam3"

# Text Prompt to guide segmentation
TEXT_PROMPT = "orange spot"

# Valid file extensions
VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def run_inference():
    print(f"=== SAM3 Text-Prompt Inference ===")
    print(f"Device: {DEVICE}")
    print(f"Prompt: '{TEXT_PROMPT}'")
    
    # Check Input
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load SAM3 Model ---
    print("Loading SAM3 model (might download weights if first time)...")
    try:
        model = build_sam3_image_model(device=DEVICE)
        processor = Sam3Processor(model)
    except Exception as e:
        print(f"Error initializing SAM3: {e}")
        print("Tip: Make sure you have run 'huggingface-cli login'")
        return

    # --- 2. Collect Images ---
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VALID_EXT)
    ]
    print(f"Found {len(image_files)} images to process.")

    # --- 3. Run Inference Loop ---
    processed_count = 0
    
    for fname in tqdm(image_files, desc="Segmenting"):
        img_path = os.path.join(INPUT_DIR, fname)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping unreadable file {fname}: {e}")
            continue

        # Set image in processor
        state = processor.set_image(image)

        # Apply text prompt
        output = processor.set_text_prompt(
            state=state,
            prompt=TEXT_PROMPT
        )

        masks = output.get("masks", None)

        if masks is None or len(masks) == 0:
            # Create empty mask if nothing found
            # (H, W) matches image size
            width, height = image.size
            combined = np.zeros((height, width), dtype=np.uint8)
        else:
            # Process masks
            masks_np = []
            for m in masks:
                m_np = m.detach().cpu().numpy()
                
                # Squeeze dimensions: (1, 1, H, W) -> (H, W) or similar
                m_np = np.squeeze(m_np)
                
                # Binarize
                m_np = (m_np > 0).astype(np.uint8)
                masks_np.append(m_np)

            # Combine all predicted masks (Union)
            combined = np.zeros_like(masks_np[0], dtype=np.uint8)
            for m in masks_np:
                combined = np.maximum(combined, m)

        # Save Binary Mask (0 or 255)
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        Image.fromarray(combined * 255).save(out_path)
        processed_count += 1

    print(f"✅ Inference complete. Processed {processed_count} images.")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
