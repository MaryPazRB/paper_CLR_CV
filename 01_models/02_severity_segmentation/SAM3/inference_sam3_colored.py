"""
SAM3 Colored Inference Script for Lesion Segmentation
=====================================================

This script runs inference using the SAM3 model with TWO text prompts:
1.  **Leaf Prompt**: Segments the entire leaf.
2.  **Disease Prompt**: Segments the disease/lesions.

Features:
-   Generates a TRANSPARENT PNG image with custom colors.
-   Calculates **Severity %** (Disease Area / Total Leaf Area).
-   Renames input filenames to include severity (e.g., `image_sev-15.50.png`).
-   Saves a summary Excel report (`severity_report.xlsx`).

Usage:
    -   Define `PROMPT_LEAF` (e.g., "leaf").
    -   Define `PROMPT_DISEASE` (e.g., "orange spot", "rust", "lesion").
"""

import os
import pandas as pd
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
INPUT_DIR = r"C:\Users\maryp\Downloads\Imagens selecionadas - paz\Imagens selecionadas - paz"

# Output Directory (Where masks will be saved)
OUTPUT_DIR = "C:/Users/maryp/Downloads/inference_sam3_colored"

# Text Prompts
PROMPT_LEAF = "leaf"          # Prompt to segment the whole leaf
PROMPT_DISEASE = "orange spot" # Prompt to segment the disease

# Colors (RGBA)
# Hex #1B3421 -> RGB (27, 52, 33)
COLOR_HEALTHY = (27, 52, 33, 255) 

# Hex #FDBD4F -> RGB (253, 189, 79)
COLOR_DISEASE = (253, 189, 79, 255)

# Valid file extensions
VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def get_mask_from_prompt(processor, state, prompt, image_size):
    """
    Runs inference for a single prompt and returns a binary mask (H, W).
    """
    output = processor.set_text_prompt(
        state=state,
        prompt=prompt
    )

    masks = output.get("masks", None)
    width, height = image_size

    if masks is None or len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Process masks
    masks_np = []
    for m in masks:
        m_np = m.detach().cpu().numpy()
        m_np = np.squeeze(m_np) # (H, W)
        m_np = (m_np > 0).astype(np.uint8)
        masks_np.append(m_np)

    # Union of all masks for this prompt
    combined = np.zeros_like(masks_np[0], dtype=np.uint8)
    for m in masks_np:
        combined = np.maximum(combined, m)
        
    return combined

def run_inference():
    print(f"=== SAM3 Colored Inference & Severity Calculation ===")
    print(f"Device: {DEVICE}")
    print(f"Leaf Prompt: '{PROMPT_LEAF}'")
    print(f"Disease Prompt: '{PROMPT_DISEASE}'")
    
    # Check Input
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load SAM3 Model ---
    print("Loading SAM3 model...")
    try:
        model = build_sam3_image_model(device=DEVICE)
        processor = Sam3Processor(model)
    except Exception as e:
        print(f"Error initializing SAM3: {e}")
        return

    # --- 2. Collect Images ---
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VALID_EXT)
    ]
    print(f"Found {len(image_files)} images to process.")

    # --- 3. Run Inference Loop ---
    results_data = [] # List to store data for Excel report
    processed_count = 0
    
    for fname in tqdm(image_files, desc="Segmenting"):
        img_path = os.path.join(INPUT_DIR, fname)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping unreadable file {fname}: {e}")
            continue

        width, height = image.size

        # Set image in processor (done once per image)
        state = processor.set_image(image)

        # A. Get Leaf Mask
        mask_leaf = get_mask_from_prompt(processor, state, PROMPT_LEAF, (width, height))

        # B. Get Disease Mask
        mask_disease = get_mask_from_prompt(processor, state, PROMPT_DISEASE, (width, height))

        # --- 4. Calculate Severity ---
        # Total Leaf Area = Union of Leaf Mask OR Disease Mask
        # (This ensures that even if 'leaf' prompt missed the disease spot, we count it as part of the leaf)
        total_leaf_mask = np.maximum(mask_leaf, mask_disease)
        
        total_pixels = np.sum(total_leaf_mask)
        disease_pixels = np.sum(mask_disease)
        
        if total_pixels > 0:
            severity = (disease_pixels / total_pixels) * 100
        else:
            severity = 0.0

        # --- 5. Create Colored Image ---
        # Initialize transparent image: (H, W, 4)
        output_img = np.zeros((height, width, 4), dtype=np.uint8)

        # 1. Paint Healthy Area (Total Leaf - Disease)
        # Anything in total_leaf_mask is part of the leaf.
        # If it is NOT disease, it is healthy.
        is_disease = (mask_disease == 1)
        is_leaf_area = (total_leaf_mask == 1)
        is_healthy = is_leaf_area & (~is_disease)

        output_img[is_healthy] = COLOR_HEALTHY
        output_img[is_disease] = COLOR_DISEASE

        # --- 6. Save File with New Name ---
        base_name = os.path.splitext(fname)[0]
        new_name = f"{base_name}_sev-{severity:.2f}.png"
        out_path = os.path.join(OUTPUT_DIR, new_name)

        Image.fromarray(output_img).save(out_path)
        
        # Add to results
        results_data.append({
            "Original Name": fname,
            "New Name": new_name,
            "Severity (%)": round(severity, 2)
        })
        
        processed_count += 1

    # --- 7. Save Excel Report ---
    if results_data:
        df = pd.DataFrame(results_data)
        excel_path = os.path.join(OUTPUT_DIR, "severity_report.xlsx")
        try:
            df.to_excel(excel_path, index=False)
            print(f"Report saved to: {excel_path}")
        except Exception as e:
            print(f"Could not save Excel report: {e}")
            # Fallback to CSV if Excel fails (e.g. missing openpyxl)
            csv_path = os.path.join(OUTPUT_DIR, "severity_report.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV instead: {csv_path}")

    print(f"✅ Inference complete. Processed {processed_count} images.")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
