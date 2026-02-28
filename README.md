# 🍃 Foundation Model–Assisted Coffee Leaf Rust Severity Estimation

This repository accompanies the manuscript:

**Foundation model–assisted segmentation enables robust field-based severity estimation of coffee leaf rust**

This project presents a fully reproducible computer vision pipeline for quantitative estimation of coffee leaf rust (*Hemileia vastatrix*) severity under heterogeneous field conditions. The framework integrates object detection, lesion segmentation, pixel-based severity quantification, and concordance analysis grounded in phytopathometry principles.

The study compares classical image processing, supervised deep learning, and foundation segmentation models for lesion detection and evaluates agreement with gold-standard pixel-level annotations using Lin’s Concordance Correlation Coefficient (LCCC).

---

## 🌱 Project Overview

The methodological workflow consists of:

1. **Leaf Detection** – YOLOv8 trained using model-assisted annotations  
2. **Leaf Extraction** – Detection-guided segmentation  
3. **Lesion Segmentation** – Comparison of five approaches:
   - ImageJ thresholding
   - pliman (R package)
   - DeepLabV3+
   - Fine-tuned SAM2 (SAM_CLR)
   - Zero-shot SAM3
4. **Severity Estimation** – Pixel-based calculation:
   S (%) = Diseased Area / Leaf Area × 100


5. **Agreement Analysis** – Lin’s Concordance Correlation Coefficient between predicted and reference severity.

   The full dataset comprises:
   - 1,285 field-acquired coffee leaf images
   - 606 curated pixel-level rust lesion masks
   -    CLR_SAM_dataset: https://universe.roboflow.com/clr-zky50/sam_clr/dataset/1
   -    DL506: https://universe.roboflow.com/clr-zky50/dl506/dataset/1
   - 100 independent evaluation masks
   GoldenStandard: https://universe.roboflow.com/clr-zky50/imgtest-fvn9j/dataset/1
---

# 📂 Repository Structure

## 📁 01_models

Contains documentation describing the trained models used in this study.

⚠️ Due to GitHub file size limitations, model weights are hosted on Hugging Face.

👉 Trained models are available here:  
**[INSERT YOUR HUGGING FACE LINK HERE]**

Models include:
- YOLOv8 leaf detector
- Fine-tuned SAM2 (SAM_CLR)
- DeepLabV3+
- Configuration used for zero-shot SAM3 inference

---

## 📁 02_binary_images

Contains validation binary masks (PNG format) corresponding to segmentation outputs from each evaluated method.

These masks were used to compute:

- Intersection over Union (IoU)
- Dice coefficient
- Pixel accuracy
- Precision
- Recall
- Disease severity (%)
- Lin’s Concordance Correlation Coefficient (LCCC)

Binary mask format:
- 0 → background
- 255 → rust lesion

This folder enables independent verification of segmentation performance and severity calculations.

---

## 📁 03_analysis

Contains R scripts used to:

- Compute severity metrics
- Perform agreement and concordance analysis
- Generate all figures included in the manuscript

Main R dependencies:
- tidyverse
- epiR
- lme4
- ggplot2

This folder reproduces the statistical analysis pipeline described in the paper.

---

# 🔬 Reproducibility

This repository provides:

- Validation segmentation outputs
- Statistical analysis scripts
- Model documentation
- External links to trained weights

Together, these components allow full reproducibility of segmentation metrics and severity agreement results reported in the manuscript.

---

# 🤖 Model Hosting

All trained model weights are hosted on Hugging Face:

👉 https://huggingface.co/MaryPazRB/Paper_CLR_CV

This ensures accessibility without exceeding GitHub file size limits.



---

# 📜 License

- Code: MIT License  
- Binary masks and annotations: CC-BY 4.0  

---

For questions or collaboration inquiries, please open an issue or contact the corresponding author.
