Coffee Leaf Rust (CLR) Severity Analysis Pipeline
=================================================

This repository contains the code for the instance segmentation and severity estimation pipeline described in "Deep learning–based segmentation of coffee leaf rust to support field severity estimation and sample size determination".

SYSTEM REQUIREMENTS
===================
*   GPU: A CUDA-capable GPU (NVIDIA) with at least 8GB VRAM is HIGHLY RECOMMENDED.
*   OS: Windows 10/11 or Linux.
*   Python: 3.8+

INSTALLATION
============
Run the following commands to install dependencies:

pip install torch torchvision numpy opencv-python pandas ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install autodistill autodistill-grounded-sam autodistill-yolov8
pip install huggingface_hub

# For SAM2 (Required for training_severity.py):
# Follow instructions at https://github.com/facebookresearch/sam2

PROJECT STRUCTURE
=================
The project is organized into three main folders corresponding to the analysis phases:

01_leaf_extraction/
    Contains codes for detecting and extracting individual leaves from field images.
    - training_leaves.py: Labels raw images and trains a YOLOv8 model.
    - extract_leaves.py: Uses YOLOv8 + SAM to crop leaves (transparent background).

02_severity_segmentation/
    Contains codes for segmenting diseased areas on the extracted leaves.
    - SAM2/
        - training_severity.py: Fine-tunes SAM2 for severity segmentation.
        - inference_clr.py: Runs inference using a fine-tuned SAM model.
    - deeplab/
        - training_deeplabv3.py: Trains a DeepLabV3+ model.
        - inference_deeplabv3.py: Runs inference using DeepLabV3+.
    - SAM3/
        - inference_sam3.py: Runs inference using SAM3 with text prompts (e.g. "yellow spot").
          (Requires Hugging Face Login)

03_trained_models/
    Contains pre-trained models developed by our lab.
    You can use these models directly with the inference scripts in folder 02 without training your own.

root/
    - sev_calculation.py: Utility to calculate severity % from binary masks.


USAGE PIPELINE
==============

PHASE 1: LEAF EXTRACTION
------------------------
1.  Run `01_leaf_extraction/extract_leaves.py`.
    Input: Folder of raw field images.
    Output: Folder of individual leaf PNGs.

PHASE 2: DISEASE SEGMENTATION
-----------------------------
Choose ONE of the available models to generate disease masks from the extracted leaves.

Option A: SAM2 (Fine-tuned)
    Run `02_severity_segmentation/inference_clr.py`.

Option B: DeepLabV3+
    Run `02_severity_segmentation/deeplab/inference_deeplabv3.py`.

Option C: SAM3 (Text Prompt)
    Run `02_severity_segmentation/SAM3/inference_sam3.py`.
    * Note: Configure the "TEXT_PROMPT" variable inside the script (e.g., "yellow spot").

* TIP: You can use the pre-trained weights located in `03_trained_models/` by pointing the `CHECKPOINT_PATH` variable in these scripts to the desired model file.

PHASE 3: SEVERITY CALCULATION
-----------------------------
1.  Run `sev_calculation.py`.
    Input: 
      - LEAF_FOLDER: Path to extracted leaves (from Phase 1).
      - MASK_FOLDER: Path to generated masks (from Phase 2).
    Output: CSV/Excel file with Severity % for each leaf.

=================================================
IMPORTANT: Configuration
Before running any script, open it in a code editor and update the CONFIGURATION section at the top. You must set the paths to your local Input/Output folders and Model Checkpoints.
=================================================
