"""
Traffic Signal Car Color Detection & People Counter
=====================================================

Trains a custom YOLOv8 detector (car + person) on your YOLO-format dataset,
trains a custom CNN color classifier on your labeled car-color data, then
runs a combined pipeline that:

  - Detects cars and people in an image
  - Classifies each detected car's color
  - Draws a RED rectangle around BLUE cars, a BLUE rectangle around all
    other colored cars, and a GREEN rectangle around people
  - Counts and reports the number of cars and people

A Gradio GUI (image upload/preview + annotated output) is included so you
don't need a separate front end.

Install dependencies first:
    pip install ultralytics gradio opencv-python-headless torch torchvision pandas pyyaml pillow matplotlib

Usage
-----
    python car_color_detection.py --train-detector          # train YOLO car/person detector
    python car_color_detection.py --train-color              # train car color classifier
    python car_color_detection.py --image path/to/photo.jpg  # run pipeline on one image, save result
    python car_color_detection.py --gui                      # launch the Gradio GUI
    python car_color_detection.py --train-detector --train-color --gui   # do it all in one go

Dataset layout expected
------------------------
Detection dataset (YOLO format, class 0 = car, class 1 = person):

    detection_dataset/
    ├── images/{train,val}/*.jpg
    └── labels/{train,val}/*.txt

Color dataset — either:

  A) CSV manifest (COLOR_LABEL_SOURCE = "csv"):
       color_dataset/
       ├── images/            (cropped car images)
       └── labels.csv         (columns: filename,color)

  B) Folder-per-class (COLOR_LABEL_SOURCE = "folder"):
       color_dataset/
       ├── blue/*.jpg
       ├── red/*.jpg
       ├── black/*.jpg
       └── ...
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

# =====================================================================
# Configuration -- edit these to match your dataset / preferences
# =====================================================================

# ----- Detection dataset (YOLO format: car + person) -----
DETECTION_DATA_ROOT = "detection_dataset"          # <-- change to your path
CLASS_NAMES = ["car", "person"]                    # class_id 0 -> car, 1 -> person

# ----- Color dataset -----
COLOR_LABEL_SOURCE = "folder"                          # "csv" or "folder"
COLOR_DATASET_ROOT = "color_dataset/train"                # <-- change to your path
COLOR_CSV_PATH = os.path.join(COLOR_DATASET_ROOT, "labels.csv")   # used if COLOR_LABEL_SOURCE == "csv"
COLOR_IMAGES_DIR = os.path.join(COLOR_DATASET_ROOT, "images")     # used if COLOR_LABEL_SOURCE == "csv"

# ----- Training hyperparameters -----
DETECTION_EPOCHS = 60
DETECTION_IMGSZ = 640
DETECTION_BATCH = 16
DETECTION_MODEL_BASE = "yolov8n.pt"   # nano = fastest; yolov8s.pt / yolov8m.pt for more accuracy

COLOR_EPOCHS = 20
COLOR_IMG_SIZE = 96
COLOR_BATCH = 32
COLOR_LR = 1e-3

# ----- Output paths -----
RUNS_DIR = "runs_detect"
DET_WEIGHTS_PATH = os.path.join(
    "runs",
    "detect",
    "runs_detect",
    "car_person_detector",
    "weights",
    "best.pt"
)
COLOR_MODEL_PATH = "car_color_classifier.pt"

# The color that gets a RED rectangle (all other predicted colors get BLUE)
TARGET_COLOR = "blue"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAR_CLASS_ID = CLASS_NAMES.index("car")
PERSON_CLASS_ID = CLASS_NAMES.index("person")

COLOR_EVAL_TF = transforms.Compose([
    transforms.Resize((COLOR_IMG_SIZE, COLOR_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

COLOR_TRAIN_TF = transforms.Compose([
    transforms.Resize((COLOR_IMG_SIZE, COLOR_IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =====================================================================
# 1. Detector training (YOLOv8, car + person)
# =====================================================================

def build_data_yaml():
    from ultralytics import YOLO  # noqa: F401  (import check only)

    data_yaml = {
        "path": os.path.abspath(DETECTION_DATA_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    os.makedirs(DETECTION_DATA_ROOT, exist_ok=True)
    data_yaml_path = os.path.join(DETECTION_DATA_ROOT, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print("Wrote", data_yaml_path)
    return data_yaml_path


def train_detector():
    from ultralytics import YOLO

    data_yaml_path = build_data_yaml()
    model = YOLO(DETECTION_MODEL_BASE)
    model.train(
        data=data_yaml_path,
        epochs=DETECTION_EPOCHS,
        imgsz=DETECTION_IMGSZ,
        batch=DETECTION_BATCH,
        project=RUNS_DIR,
        name="car_person_detector",
        exist_ok=True,
    )
    print("Best detector weights saved at:", DET_WEIGHTS_PATH)      

    trained = YOLO(DET_WEIGHTS_PATH)
    metrics = trained.val(data=data_yaml_path)
    print(metrics)


# =====================================================================
# 2. Color classifier training (MobileNetV2 transfer learning)
# =====================================================================

class CsvColorDataset(Dataset):
    """Reads a CSV manifest: columns = filename,color"""

    def __init__(self, csv_path, images_dir, transform, class_to_idx=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        classes = sorted(self.df["color"].str.lower().unique().tolist())
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(classes)}
        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[str(row["color"]).lower()]
        return img, label


def build_color_datasets():
    if COLOR_LABEL_SOURCE == "csv":
        full_train = CsvColorDataset(COLOR_CSV_PATH, COLOR_IMAGES_DIR, COLOR_TRAIN_TF)
        classes, class_to_idx = full_train.classes, full_train.class_to_idx
        full_eval = CsvColorDataset(COLOR_CSV_PATH, COLOR_IMAGES_DIR, COLOR_EVAL_TF, class_to_idx)
    elif COLOR_LABEL_SOURCE == "folder":
        from torchvision.datasets import ImageFolder
        full_train = ImageFolder(COLOR_DATASET_ROOT, transform=COLOR_TRAIN_TF)
        classes, class_to_idx = full_train.classes, full_train.class_to_idx
        full_eval = ImageFolder(COLOR_DATASET_ROOT, transform=COLOR_EVAL_TF)
    else:
        raise ValueError("COLOR_LABEL_SOURCE must be 'csv' or 'folder'")

    n = len(full_train)
    n_val = max(1, int(0.15 * n))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(42)
    train_subset, _ = random_split(full_train, [n_train, n_val], generator=gen)
    gen = torch.Generator().manual_seed(42)
    _, val_subset = random_split(full_eval, [n_train, n_val], generator=gen)
    return train_subset, val_subset, classes


def build_color_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False  # freeze backbone, fine-tune head only
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def train_color_classifier():
    train_ds, val_ds, classes = build_color_datasets()
    print("Color classes found:", classes)

    train_loader = DataLoader(train_ds, batch_size=COLOR_BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=COLOR_BATCH, shuffle=False, num_workers=2)

    model = build_color_model(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=COLOR_LR)

    best_val_acc = 0.0
    for epoch in range(1, COLOR_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, train=False)
        print(f"Epoch {epoch:02d}/{COLOR_EPOCHS} | train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"| val_loss={val_loss:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, COLOR_MODEL_PATH)

    print("Best val accuracy:", best_val_acc)
    print("Saved color classifier to:", COLOR_MODEL_PATH)


# =====================================================================
# 3. Combined inference pipeline
# =====================================================================

_det_model = None
_color_model = None
_color_classes = None


def get_detector():
    global _det_model
    if _det_model is None:
        from ultralytics import YOLO
        if not os.path.exists(DET_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Detector weights not found at {DET_WEIGHTS_PATH}. Run with --train-detector first."
            )
        _det_model = YOLO(DET_WEIGHTS_PATH)
    return _det_model


def get_color_model():
    global _color_model, _color_classes
    if _color_model is None:
        if not os.path.exists(COLOR_MODEL_PATH):
            raise FileNotFoundError(
                f"Color classifier not found at {COLOR_MODEL_PATH}. Run with --train-color first."
            )
        checkpoint = torch.load(COLOR_MODEL_PATH, map_location=DEVICE)
        _color_classes = checkpoint["classes"]
        _color_model = build_color_model(len(_color_classes))
        _color_model.load_state_dict(checkpoint["model_state"])
        _color_model.eval()
    return _color_model, _color_classes


def classify_car_color(crop_bgr):
    """crop_bgr: numpy array (H, W, 3) in BGR. Returns predicted color string."""
    model, classes = get_color_model()
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = COLOR_EVAL_TF(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        pred_idx = logits.argmax(1).item()
    return classes[pred_idx]


def run_pipeline(image_bgr, conf=0.35):
    """Runs detection + color classification, returns (annotated_image_bgr, summary_dict)."""
    detector = get_detector()
    results = detector.predict(image_bgr, conf=conf, verbose=False)[0]

    annotated = image_bgr.copy()
    car_count = 0
    person_count = 0
    color_breakdown = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if cls_id == CAR_CLASS_ID:
            car_count += 1
            crop = image_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue
            color = classify_car_color(crop)
            color_breakdown[color] = color_breakdown.get(color, 0) + 1

            # Required rule: blue cars -> RED rectangle, all other colors -> BLUE rectangle
            rect_color = (0, 0, 255) if color.lower() == TARGET_COLOR.lower() else (255, 0, 0)  # BGR

            cv2.rectangle(annotated, (x1, y1), (x2, y2), rect_color, 2)
            cv2.putText(annotated, f"car:{color}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

        elif cls_id == PERSON_CLASS_ID:
            person_count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green
            cv2.putText(annotated, "person", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    summary_text = f"Cars: {car_count}   People: {person_count}"
    text_w = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    cv2.rectangle(annotated, (5, 5), (10 + text_w, 40), (0, 0, 0), -1)
    cv2.putText(annotated, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    summary = {"car_count": car_count, "person_count": person_count, "color_breakdown": color_breakdown}
    return annotated, summary


def run_on_image_file(image_path, output_path="annotated_output.jpg"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    annotated, summary = run_pipeline(img_bgr)
    cv2.imwrite(output_path, annotated)
    print("Summary:", summary)
    print("Annotated image saved to:", output_path)


# =====================================================================
# 4. GUI (Gradio) -- image upload/preview + annotated result
# =====================================================================

def launch_gui():
    import gradio as gr

    def gradio_infer(input_image):
        if input_image is None:
            return None, "Upload an image to begin."
        img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        annotated_bgr, summary = run_pipeline(img_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        breakdown_lines = "\n".join(f"  - {c}: {n}" for c, n in summary["color_breakdown"].items())
        breakdown_lines = breakdown_lines or "  - (no cars detected)"
        report = (
            f"Cars detected: {summary['car_count']}\n"
            f"People detected: {summary['person_count']}\n\n"
            f"Color breakdown:\n{breakdown_lines}\n\n"
            f"Legend: red box = blue car | blue box = other-color car | green box = person"
        )
        return annotated_rgb, report

    with gr.Blocks(title="Traffic Signal Car Color Detector") as demo:
        gr.Markdown("# Car Color Detection & People Counter\nUpload a traffic-signal photo to preview detection results.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Input image (preview)")
            out_img = gr.Image(type="numpy", label="Annotated output")
        out_text = gr.Textbox(label="Summary", lines=8)
        run_btn = gr.Button("Run detection", variant="primary")
        run_btn.click(fn=gradio_infer, inputs=inp, outputs=[out_img, out_text])
        inp.change(fn=gradio_infer, inputs=inp, outputs=[out_img, out_text])

    demo.launch(share=True)


# =====================================================================
# CLI entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Car color detection & people counting pipeline")
    parser.add_argument("--train-detector", action="store_true", help="Train the YOLOv8 car/person detector")
    parser.add_argument("--train-color", action="store_true", help="Train the car color classifier")
    parser.add_argument("--image", type=str, default=None, help="Run the pipeline on a single image and save the result")
    parser.add_argument("--output", type=str, default="annotated_output.jpg", help="Output path for --image")
    parser.add_argument("--gui", action="store_true", help="Launch the Gradio GUI")
    args = parser.parse_args()

    if args.train_detector:
        train_detector()
    if args.train_color:
        train_color_classifier()
    if args.image:
        run_on_image_file(args.image, args.output)
    if args.gui:
        launch_gui()

    if not any([args.train_detector, args.train_color, args.image, args.gui]):
        parser.print_help()


if __name__ == "__main__":
    main()
