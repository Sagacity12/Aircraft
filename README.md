# ✈️ Aircraft Damage Detector

Fine-tuned VGG16 model trained to detect and classify aircraft structural damage from images. Integrates a BLIP image captioning layer to generate natural language descriptions of detected damage. Built with TensorFlow/Keras for classification and HuggingFace Transformers for vision-language generation.

---

## 📌 Overview

This project combines **transfer learning** and **vision-language modelling** to build an end-to-end aircraft damage detection pipeline. Given an image of an aircraft, the system:

1. Classifies whether the aircraft is **damaged or undamaged**
2. Generates a **natural language caption or summary** describing the damage

---

## 🧠 Model Architecture

### Classifier — VGG16 + Custom Head

```
Input Image (128×128×3)
        ↓
   VGG16 Base (frozen, pretrained on ImageNet)
        ↓
     Flatten
        ↓
  Dense(512) + ReLU + Dropout(0.3)
        ↓
  Dense(512) + ReLU + Dropout(0.3)
        ↓
   Dense(1) + Sigmoid
        ↓
  Binary Output (damaged / undamaged)
```

### Captioning — BLIP

```
Input Image
      ↓
 BlipProcessor
      ↓
 BlipForConditionalGeneration
      ↓
 Natural Language Caption / Summary
```

---

## 📁 Project Structure

```
Aircraft Damage Using Pretrained Model/
│
├── main.py                        # Main training and inference script
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
└── aircraft_damage_dataset_v1/    # Auto-downloaded dataset
    ├── train/
    │   ├── damaged/
    │   └── undamaged/
    ├── valid/
    │   ├── damaged/
    │   └── undamaged/
    └── test/
        ├── damaged/
        └── undamaged/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 (recommended)
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aircraft-damage-detector.git
cd aircraft-damage-detector

# Create and activate virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

The script will automatically download and extract the dataset, train the model, evaluate it, and run BLIP captioning on sample images.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Model training and Keras layers |
| `keras` | High-level neural network API |
| `torch` | Required by BLIP/HuggingFace |
| `transformers==4.35.0` | BLIP processor and model |
| `Pillow` | Image loading and processing |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting training curves |

Install all at once:

```bash
pip install tensorflow torch transformers==4.35.0 Pillow numpy matplotlib
```

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Base model | VGG16 (ImageNet weights) |
| Image size | 128 × 128 |
| Batch size | 32 |
| Epochs | 10 |
| Optimiser | Adam (lr=0.0001) |
| Loss | Binary Crossentropy |
| Metric | Accuracy |

### Data Augmentation (train only)

- Horizontal flip
- Vertical flip
- Rotation (±20°)
- Rescale (1/255)

---

## 🖼️ BLIP Image Captioning

The `BlipCaptionSummaryLayer` is a custom Keras layer that wraps the BLIP model using `tf.py_function` to bridge PyTorch and TensorFlow.

Supports two tasks:

```python
# Generate a short caption
caption = blip_layer(image_path, tf.constant("caption"))

# Generate a detailed summary
summary = blip_layer(image_path, tf.constant("summary"))
```

---

## 📈 Results

After training, the following plots are generated:

- Training vs Validation Loss curve
- Training vs Validation Accuracy curve
- Sample prediction with true vs predicted label
- BLIP caption and summary for test images

---

## 🏷️ Topics

`transfer-learning` `binary-classification` `vgg16` `pretrained-models` `feature-extraction`
`data-augmentation` `image-preprocessing` `blip-image-captioning` `huggingface-transformers`
`custom-keras-layers` `vision-language-models` `pytorch-tensorflow-bridge` `deep-learning`
`computer-vision` `tensorflow` `keras`

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset provided by [IBM Skills Network](https://skills.network)
- VGG16 pretrained weights from [ImageNet](https://www.image-net.org/)
- BLIP model by [Salesforce Research](https://github.com/salesforce/BLIP)
- HuggingFace [Transformers](https://huggingface.co/transformers)
