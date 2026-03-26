# 📄 Data Card — CLR_SAM and DL_506 (Coffee Leaf Rust Segmentation Models)

---

## 1. Summary

**Name:** CLR_SAM  
**Version:** v1.0  
**Type:** Image dataset + segmentation model  
**Task:** Pixel-level segmentation for disease severity estimation  

**Name:** DL_506  
**Version:** v1.0  
**Type:** Image dataset + segmentation model  
**Task:** Pixel-level segmentation for disease severity estimation  

**Description:**  
CLR_SAM is a fine-tuned segmentation model based on the Segment Anything Model (SAM2), designed to identify and quantify rust lesions caused by *Hemileia vastatrix* in *Coffea arabica* leaves. The model enables automated estimation of disease severity under heterogeneous field conditions using RGB images.

DL_506 is a deep learning segmentation model based on DeepLabV3+, designed to identify and quantify rust lesions caused by *Hemileia vastatrix* in *Coffea arabica* leaves. The model enables automated estimation of disease severity under heterogeneous field conditions using RGB images.

**Primary Outputs:**
- Binary segmentation masks (diseased vs healthy tissue)
- Disease severity (% of diseased leaf area)

---

## 2. Authorship & Ownership

- **Creators:** Mary Paz Romero Benavides et al.  
- **Affiliation:** Universidade Federal de Viçosa (UFV), EPAMIG Sudeste  
- **Contact:** emdelponte@ufv.br  
- **Funding:** CAPES, FAPEMIG, CNPQ  

---

## 3. Dataset Overview

### Data Type
- Natural phenomena (plant disease)
- Plant data (coffee leaves)
- No human or personal data included

### Dataset Summary

| Attribute | Value |
|----------|------|
| Total images | 1,285 |
| Annotated masks | 606 |
| Manual annotations | 100 |
| Expanded dataset | 506 |
| Evaluation set | 100 |
| Resolution | 1024 × 1024 |
| Task | Binary segmentation |

### Content Description

The dataset consists of RGB images of coffee leaves with varying levels of coffee leaf rust severity. Images were collected under:

- Field conditions (natural light, complex backgrounds)
- Controlled conditions (uniform lighting, black background)
- Branch-level images (multi-leaf, real-world conditions)

Annotations include:
- Pixel-level lesion masks
- Leaf segmentation masks
- Derived severity values (%)

---

## 4. Motivation & Intended Use

### Motivation

Accurate estimation of coffee leaf rust severity is critical for:
- Epidemiological studies
- Disease monitoring
- Resistance screening
- Agricultural decision-making

Visual assessment is subjective and prone to error. CLR_SAM and DL_506 aim to:
- Reduce observer bias
- Improve reproducibility
- Enable scalable and automated severity estimation

### Intended Use

- Automated disease severity estimation
- Research in phytopathometry
- Benchmarking segmentation models
- Precision agriculture applications

### Out-of-Scope Use

- Other plant diseases without validation
- Other crops without adaptation
- Clinical or human-related applications

---

## 5. Data Provenance & Collection

### Locations

Data were collected in Minas Gerais, Brazil:
- Araponga
- Oratórios
- Leopoldina
- Ervália  

### Collection Method

- RGB images captured using smartphones
- Natural and controlled lighting conditions
- Leaves sampled from susceptible plants (>1 year old)

### Annotation Process

- Manual expert annotation
- Model-assisted annotation (human-in-the-loop)
- COCO format masks

### Preprocessing

- Image resizing (1024 × 1024)
- Binary mask generation
- Leaf extraction using YOLOv8 + SAM2

---

## 6. Model Development

### Pipeline

1. Leaf detection (YOLOv8)  
2. Leaf segmentation (SAM2)  
3. Lesion segmentation (CLR_SAM or DL_506)  
4. Severity estimation  

### Severity Calculation

S (%) = (Diseased Area / Leaf Area) × 100  

---

### Model Details

#### **CLR_SAM (SAM2 Fine-Tuned Model)**

- Base model: SAM2 (Hiera Tiny backbone)
- Framework: PyTorch + SAM2
- Input size: up to 1024 px (aspect-ratio preserved)
- Training dataset: 100 manually annotated images
- Data split: 80% training / 20% test

**Training configuration:**
- Training steps: 6,000  
- Effective batch size: ~8 (gradient accumulation = 8)  
- Optimizer: AdamW  
- Learning rate: 5e-5  
- Weight decay: 1e-4  
- LR scheduler: StepLR  
  - Step size: 2,000  
  - Gamma: 0.6  
- Mixed precision: Yes (AMP + GradScaler)  
- Gradient clipping: max norm = 1.0  

**Loss function:**
- Binary Cross-Entropy (segmentation)
- IoU-based score regression loss  
- Final loss: segmentation loss + 0.05 × score loss  

**Prompting strategy:**
- Up to 5 point prompts per image  
- Points sampled from eroded lesion regions  
- All prompts labeled as foreground  

**Training strategy:**
- Image encoder frozen  
- Fine-tuned modules:
  - Mask decoder  
  - Prompt encoder  
- Random sample per step (no fixed batching)  

**Preprocessing:**
- Resize with aspect ratio preservation  
- Binary mask conversion  
- Morphological erosion for point selection  

**Checkpointing:**
- Model saved every 500 steps  

---

#### **DL_506 (DeepLabV3+)**

- Base model: DeepLabV3+ (encoder–decoder CNN)
- Encoder: ResNet-50 (ImageNet pretrained)
- Input size: 512 × 512
- Training dataset: 506 annotated images

**Training configuration:**
- Epochs: 25  
- Batch size: 8  
- Optimizer: AdamW  
- Learning rate: 1e-4  

**Loss function:**
- Dice Loss (binary)
- Binary Cross-Entropy (BCEWithLogitsLoss)
- Combined loss: Dice + BCE  

**Data augmentation:**
- Horizontal flip (p = 0.5)  
- Vertical flip (p = 0.5)  

**Normalization:**
- Mean: [0.485, 0.456, 0.406]  
- Std: [0.229, 0.224, 0.225]  

**Training strategy:**
- Fully supervised segmentation  
- Binary classification (lesion vs background)  
- Best model selected via validation loss  

**Implementation details:**
- Framework: PyTorch + segmentation_models_pytorch  
- Checkpointing: best and final models saved  
- Hardware: GPU recommended  

---

## 7. Evaluation

### Pixel-Level Performance

- Dice ≈ 0.91  
- IoU ≈ 0.83  
- Precision ≈ 0.93  
- Recall ≈ 0.94  

### Leaf-Level Agreement

- High concordance with reference severity  
- Reduced bias compared to classical methods  

---

## 8. Risks & Limitations

### Domain Shift

Performance may decrease under:
- Different lighting conditions  
- New geographic regions  
- Different cultivars  

**Mitigation:**  
Training includes diverse field conditions; fine-tuning recommended.

---

### Annotation Bias

Manual annotations may introduce subjectivity.

**Mitigation:**  
Expert validation and independent evaluation dataset.

---

### Segmentation Errors

Confusion may occur with:
- Leaf senescence  
- Shadows or background artifacts  

---

### Computational Requirements

- GPU recommended for training and inference  

---

## 9. Ethical Considerations

- No personal or sensitive human data  
- Supports agricultural research  
- Low risk of misuse  

---

## 10. Access & Maintenance

- **Repository:** https://github.com/MaryPazRB/paper_CLR_CV  
- **License:** MIT  
- **Maintenance:** Actively maintained  

### Future Improvements

- Expanded annotation dataset  
- Multi-disease support  
- Mobile deployment optimization  

---

## 11. Compatibility with Other Data

### Compatible With

- Other plant disease datasets  
- Agronomic and environmental datasets  

### Use with Caution

- Different imaging modalities (e.g., hyperspectral)  
