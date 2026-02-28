import os
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# USER CONFIGURATION
# ==========================================
# Path to your YOLO model file (.pt)
MODEL_PATH = r"G:\.shortcut-targets-by-id\1ig9YbfY4r3ujzEoaN6_wIKnZPNNel2ma\Mary_Paz\Projeto mestrado cafe (1)\Papers\Article_2_models\03_trained_models\clr_YOLOV8.pt"

# Path to the image file OR the folder containing images you want to process
SOURCE_PATH = r"G:\My Drive\Organic_coffee_sev - Copy\total\IMG_318.jpg"

# Folder where the results will be saved
OUTPUT_DIR = r"G:\.shortcut-targets-by-id\1ig9YbfY4r3ujzEoaN6_wIKnZPNNel2ma\Mary_Paz\Projeto mestrado cafe (1)\Papers\Article_2_models\03_trained_models\figure_example1"

# Confidence threshold for detection (0.0 to 1.0)
CONF_THRESHOLD = 0.5
# ==========================================

def process_images(model_path, source_path, output_dir, conf_threshold):
    """
    Run YOLO inference on images.
    """
    # 1. Load the model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # 2. Resolve paths
    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 3. Collect images
    images_to_process = []
    if source.is_file():
        images_to_process.append(source)
    elif source.is_dir():
        # Add common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            images_to_process.extend(source.glob(ext))
            # Also check uppercase extensions
            images_to_process.extend(source.glob(ext.upper()))
    else:
        print(f"Error: Source path '{source}' does not exist.")
        return

    if not images_to_process:
        print(f"No images found at {source}")
        return

    print(f"Processing {len(images_to_process)} images...")

    # 4. Run inference
    for img_path in images_to_process:
        print(f"Predicting: {img_path.name}")
        try:
            model.predict(
                source=str(img_path),
                save=True,
                project=str(output),
                name="",  # Don't create valid/train subfolders, just save to project dir usually
                exist_ok=True, # Allow overwriting/appending to existing project dir
                conf=conf_threshold
            )
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")

    print(f"Done! Results saved to {output.resolve()}")

def main():
    print("Starting Leaf Extraction...")
    print(f"Model: {MODEL_PATH}")
    print(f"Source: {SOURCE_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    process_images(MODEL_PATH, SOURCE_PATH, OUTPUT_DIR, CONF_THRESHOLD)

if __name__ == "__main__":
    main()
