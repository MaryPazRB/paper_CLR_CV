import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ------------------------
# Config
# ------------------------
IMAGE_DIR = r"G:\.shortcut-targets-by-id\1ig9YbfY4r3ujzEoaN6_wIKnZPNNel2ma\Mary_Paz\Projeto mestrado cafe (1)\P1\chosentrainer2"
OUTPUT_DIR = r"G:\.shortcut-targets-by-id\1ig9YbfY4r3ujzEoaN6_wIKnZPNNel2ma\Mary_Paz\Projeto mestrado cafe (1)\P1\chosentrainer2_SAM3"
TEXT_PROMPT = "orange spot"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------
# Load SAM3
# ------------------------
model = build_sam3_image_model(device=DEVICE)
processor = Sam3Processor(model)


# ------------------------
# Collect images
# ------------------------
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(VALID_EXT)
]

print(f"Found {len(image_files)} images")


# ------------------------
# Run batch segmentation
# ------------------------
for fname in tqdm(image_files, desc="Text Prompt Segment"):
    img_path = os.path.join(IMAGE_DIR, fname)

    image = Image.open(img_path).convert("RGB")

    # Set image
    state = processor.set_image(image)

    # Apply text prompt
    output = processor.set_text_prompt(
        state=state,
        prompt=TEXT_PROMPT
    )

    masks = output.get("masks", None)

    if masks is None or len(masks) == 0:
        print(f"⚠️ No masks for {fname}")
        continue

    # Convert masks to CPU numpy and squeeze dimensions
    masks_np = []
    for m in masks:
        m_np = m.detach().cpu().numpy()

        # 🔧 CRITICAL FIX: squeeze singleton dimensions
        m_np = np.squeeze(m_np)

        # Ensure binary uint8
        m_np = (m_np > 0).astype(np.uint8)
        masks_np.append(m_np)

    combined = np.zeros_like(masks_np[0], dtype=np.uint8)
    for m in masks_np:
        combined = np.maximum(combined, m)

    # Save mask (now guaranteed H x W)
    out_name = os.path.splitext(fname)[0] + ".png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    Image.fromarray(combined * 255).save(out_path)


print("✅ Text-prompt batch finished")
