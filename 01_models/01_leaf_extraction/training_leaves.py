"""
Automated Labeling and Training Pipeline for Coffee Leaf Detection
==================================================================

This script demonstrates the process of using a foundation model (GroundedSAM)
to automatically label a dataset of coffee leaves, followed by training a 
smaller, faster model (YOLOv8) on the distilled data.

Requirements:
    - autodistill
    - autodistill-grounded-sam
    - autodistill-yolov8
"""

import os
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# ================= Configuration =================
# Define your input and output directories here.
# INPUT_FOLDER: Directory containing raw images to be labeled.
# OUTPUT_FOLDER: Directory where the labeled dataset (images + YOLO labels) will be saved.
INPUT_FOLDER = "./data/raw_images" 
OUTPUT_FOLDER = "./data/labeled_dataset"

# Define the ontology for the foundation model.
# Format: {"text_prompt": "class_name"}
ONTOLOGY = CaptionOntology({
    "leaf": "coffee leaf"
})

# Training parameters
YOLO_BASE_MODEL = "yolov8n.pt"  # Use generic path or simple name for download
EPOCHS = 200
# =================================================

def main():
    print("--- 1. Initializing Foundation Model (GroundedSAM) ---")
    base_model = GroundedSAM(ontology=ONTOLOGY)

    print(f"--- 2. Auto-labeling images from {INPUT_FOLDER} ---")
    # This step generates the dataset in YOLO format at the OUTPUT_FOLDER
    base_model.label(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER
    )
    print(f"Dataset generated at: {OUTPUT_FOLDER}")

    print("--- 3. Training Target Model (YOLOv8) ---")
    target_model = YOLOv8(YOLO_BASE_MODEL)
    
    # The 'data.yaml' file is automatically created by autodistill in the output folder
    dataset_yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    
    target_model.train(
        dataset_yaml=dataset_yaml_path,
        epochs=EPOCHS
    )
    print("Training complete.")

    # --- 4. Inference Example (Optional) ---
    # Run inference on a test image to verify performance
    # test_image_path = "./data/test_image.jpg"
    # if os.path.exists(test_image_path):
    #     pred = target_model.predict(test_image_path, confidence=0.5)
    #     print("Prediction:", pred)

if __name__ == "__main__":
    main()
