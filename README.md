# 🚗 Car Color Detection & Person Counter

A computer vision project that combines **YOLOv8 object detection** and **MobileNetV2 transfer learning** to detect vehicles and people, classify detected vehicle colors, count objects, and visualize the results through an interactive **Gradio** interface.

The project demonstrates an end-to-end computer vision pipeline covering dataset preparation, model training, inference, visualization, and a simple web-based interface.

---

## ✨ Features

- 🚗 Detect vehicles in traffic-scene images
- 👤 Detect and count people
- 🎨 Classify the color of detected vehicles
- 🔴 Draw a red bounding box around **blue vehicles**
- 🔵 Draw a blue bounding box around **other detected vehicles**
- 🟢 Draw a green bounding box around **people**
- 📊 Display vehicle and people counts
- 📈 Generate a color-wise vehicle breakdown
- 🖼️ Save annotated output images
- 🌐 Interactive Gradio web interface
- 🧠 Uses transfer learning with MobileNetV2 for color classification
- ⚡ Supports CPU and CUDA-enabled PyTorch environments

---

## 🏗️ System Architecture

```text
                     Input Traffic Image
                              │
                              ▼
                    ┌──────────────────┐
                    │     YOLOv8       │
                    │  Object Detector │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
                 Vehicle            Person
                    │                 │
                    ▼                 ▼
            Crop Vehicle Image    Green Box
                    │
                    ▼
             ┌───────────────┐
             │  MobileNetV2  │
             │Color Classifier│
             └───────┬───────┘
                     │
                     ▼
               Predicted Color
                     │
             ┌───────┴────────┐
             │                │
          Blue Vehicle    Other Colors
             │                │
             ▼                ▼
          Red Box          Blue Box
             │                │
             └───────┬────────┘
                     ▼
              Annotated Image
                     │
                     ▼
             Counts & Summary
```

---

## 🧠 Models

### 1. YOLOv8 Object Detector

The project uses **YOLOv8n** as the base object-detection model and trains it on a custom YOLO-format dataset.

The detector uses two classes:

```text
0 → car
1 → person
```

The original detection dataset used a general **vehicle** class rather than a separate car class. Its labels were converted to match this project's class convention.

As a result, the current detector should be understood as a **vehicle + person detector**, with vehicles represented by the project's `car` label.

### 2. MobileNetV2 Color Classifier

The vehicle-color classifier uses **MobileNetV2 pretrained on ImageNet**.

The pretrained feature extractor is frozen and a new classification layer is trained for the vehicle-color dataset.

The classifier is trained using the folder-per-class dataset format, where each folder represents a color category.

---

## 🎯 Annotation Rules

The application applies the following visualization rules:

| Detected object | Bounding-box color |
|---|---|
| Blue vehicle | 🔴 Red |
| Other-color vehicle | 🔵 Blue |
| Person | 🟢 Green |

The color classifier predicts the vehicle color first, after which the corresponding bounding-box color is selected.

---

## 📊 Training Results

### YOLOv8 Vehicle & Person Detector

Training configuration:

| Parameter | Value |
|---|---:|
| Base model | YOLOv8n |
| Epochs | 60 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Classes | Vehicle, Person |

The trained detector achieved approximately:

**mAP@50: ~87.7% on the validation set**

Best detector weights:

```text
runs/detect/runs_detect/car_person_detector/weights/best.pt
```

### MobileNetV2 Vehicle Color Classifier

Training configuration:

| Parameter | Value |
|---|---:|
| Backbone | MobileNetV2 |
| Input size | 96 × 96 |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Backbone | Frozen |
| Training approach | Transfer learning |

The best validation accuracy achieved during the project's color-classifier training was approximately:

**73.34%**

Color-classification performance can vary significantly with lighting, reflections, vehicle size, image quality, and visually similar colors.

---

## 🔄 How the Pipeline Works

### Step 1 — Object Detection

YOLOv8 processes the complete image and identifies vehicles and people.

For every detection, the model provides a class and bounding box.

### Step 2 — Vehicle Cropping

When a vehicle is detected, its bounding-box region is cropped from the original image.

### Step 3 — Color Classification

The cropped vehicle image is converted to RGB and passed through the MobileNetV2-based classifier.

The classifier predicts its color.

### Step 4 — Visualization

The predicted color determines the vehicle's bounding-box color:

```text
Blue vehicle  → Red box
Other vehicle → Blue box
Person        → Green box
```

### Step 5 — Counting

The application maintains separate counts for:

```text
Vehicles
People
```

It also creates a color breakdown showing how many detected vehicles were assigned to each predicted color.

---

## 📁 Project Structure

```text
Car-color-detection-and-person-counter/
│
├── car_color_detection.py
├── README.md
├── .gitignore
│
├── car_color_classifier.pt
│
└── runs/
    └── detect/
        └── runs_detect/
            └── car_person_detector/
                └── weights/
                    └── best.pt
```

### Local-only data

The following directories contain large datasets and training artifacts and should remain excluded from Git:

```text
detection_dataset/
color_dataset/
dataset/
runs/
```

---

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLOv8 | Vehicle and person detection |
| PyTorch | Deep-learning framework |
| TorchVision | MobileNetV2 and image transforms |
| MobileNetV2 | Vehicle color classification |
| OpenCV | Image processing and annotation |
| Gradio | Web interface |
| NumPy | Numerical operations |
| Pandas | CSV dataset handling |
| PyYAML | YOLO dataset configuration |
| Pillow | Image loading and processing |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Apex-afk/Car-color-detection-and-person-counter.git
cd Car-color-detection-and-person-counter
```

### 2. Create the Python environment

Using Conda:

```bash
conda create -n carcolor python=3.11
conda activate carcolor
```

### 3. Install dependencies

The project was developed in Python 3.11.

```bash
pip install torch torchvision torchaudio
pip install opencv-python==4.10.0.84
pip install numpy==1.26.4
pip install ultralytics gradio pandas pyyaml pillow matplotlib
```

---

## 📦 Dataset Setup

### Detection Dataset

The detector expects a YOLO-format dataset:

```text
detection_dataset/
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
```

Each label follows the YOLO format:

```text
class_id x_center y_center width height
```

The project uses:

```text
0 = car
1 = person
```

The source dataset originally used:

```text
0 = person
1 = vehicle
```

The labels were converted during dataset preparation so they match the class order required by this project.

### Vehicle Color Dataset

The color classifier currently uses the folder-based format:

```text
color_dataset/
└── train/
    ├── black/
    ├── blue/
    ├── brown/
    ├── green/
    ├── grey/
    ├── orange/
    ├── red/
    ├── white/
    └── ...
```

Each directory represents one color class.

---

## 🏋️ Training

### Train the YOLOv8 detector

```bash
python car_color_detection.py --train-detector
```

The detector trains for 60 epochs and saves its best weights under:

```text
runs/detect/runs_detect/car_person_detector/weights/best.pt
```

### Train the vehicle-color classifier

```bash
python car_color_detection.py --train-color
```

The trained classifier is saved as:

```text
car_color_classifier.pt
```

### Train both models

```bash
python car_color_detection.py --train-detector --train-color
```

---

## 🖼️ Run on an Image

Place an image in the project directory, for example:

```text
test.jpg
```

Run:

```bash
python car_color_detection.py --image test.jpg
```

The pipeline will:

1. Detect vehicles and people
2. Classify detected vehicle colors
3. Draw bounding boxes
4. Count vehicles
5. Count people
6. Generate a color breakdown
7. Save the annotated result

Default output:

```text
annotated_output.jpg
```

A custom output path can also be provided:

```bash
python car_color_detection.py --image test.jpg --output result.jpg
```

---

## 🌐 Gradio Interface

Launch the interactive web interface:

```bash
python car_color_detection.py --gui
```

The interface allows you to:

- Upload a traffic image
- Preview the input
- Run the detection pipeline
- View the annotated output
- View vehicle and people counts
- View the color breakdown

The application uses Gradio's sharing option to provide a shareable interface URL when launched.

---

## 📋 Example Output

For a test image, the pipeline can return a summary such as:

```python
{
    "car_count": 8,
    "person_count": 1,
    "color_breakdown": {
        "white": 1,
        "orange": 1,
        "red": 1,
        "blue": 1,
        "black": 1,
        "grey": 1,
        "yellow": 1,
        "green": 1
    }
}
```

The annotated image uses:

```text
🔴 Blue vehicle
🔵 Other vehicle colors
🟢 Person
```

---

## ⚠️ Limitations

### 1. Vehicle types are not separated

The detection dataset uses a general vehicle category. Therefore, the current model does not distinguish between cars, buses, trucks, and other vehicle types.

For example:

```text
Bus   → car label
Car   → car label
Truck → car label
```

This behavior comes from the available training labels.

### 2. Color classification depends on image conditions

Predictions can be affected by:

- Low lighting
- Shadows
- Reflections
- Motion blur
- Occlusion
- Small vehicle crops
- Camera quality
- Similar colors such as grey and silver

### 3. No multi-object tracking

The current implementation performs detection independently on each image.

It does not yet use a tracking algorithm to maintain object identities across video frames. Therefore, frame-by-frame video counting should not be interpreted as unique-vehicle counting.

### 4. Missed detections cannot be color-classified

Color classification is performed only after a vehicle has been detected. If YOLO fails to detect a vehicle, the color classifier does not receive a crop for that vehicle.

---

## 🚀 Future Improvements

- 🎯 Train a detector with separate classes for car, bus, truck, motorcycle, etc.
- 🎨 Improve color classification using a larger and more balanced dataset
- 🧠 Fine-tune additional MobileNetV2 layers
- 📹 Add real-time video and webcam support
- 🔄 Add multi-object tracking
- 📈 Add traffic analytics and historical statistics
- 🚦 Add traffic-light detection
- 🗺️ Analyze traffic flow by direction or lane
- ☁️ Deploy the application online
- 📱 Create a mobile-friendly interface
- ⚡ Optimize the pipeline for real-time inference

---

## 📚 Dataset Sources

### Vehicle & Person Detection

**YOLO Person Vehicle Detection Dataset Annotated**

Kaggle:  
https://www.kaggle.com/datasets/haseebhsb/yolo-person-vehicle-detection-dataset-annotated

The dataset was converted to match the class convention used by this project.

### Vehicle Color Recognition

**VCoR — Vehicle Color Recognition Dataset**

Kaggle:  
https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset

The project uses the folder-based organization of the color dataset.

Please refer to the original dataset pages for their current licenses and attribution requirements before redistributing dataset material.

---

## 👨‍💻 Author

**Kunal Gulve**

GitHub:  
https://github.com/Apex-afk

---

## 📄 License

This project is provided for educational and research purposes.

The datasets used by the project remain subject to their respective licenses and terms. Please review the original dataset licenses before redistribution.

---

## ⭐ Project Summary

This project combines:

```text
YOLOv8
   +
MobileNetV2
   +
PyTorch
   +
OpenCV
   +
Gradio
```

to create an end-to-end computer vision system capable of detecting vehicles and people, classifying vehicle colors, counting detected objects, applying color-based annotations, and presenting the results through an interactive web interface.
