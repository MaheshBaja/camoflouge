# 🦎 Camouflaged Object Detection (COD)

An end-to-end inference application for **Camouflaged Object Detection / Segmentation** using a trained **U-Net** with a **ResNet34** encoder backbone trained on the **COD10K-v3** dataset.

---

## 📌 Project Overview

Camouflaged Object Detection (COD) aims to identify and segment objects that blend seamlessly into their surroundings due to visual patterns, textures, and colors matching the background.

### 🧠 Model & Training Specifications
- **Architecture:** U-Net
- **Backbone (Encoder):** ResNet34 (pretrained on ImageNet during training)
- **Input Image Size:** 352 × 352 pixels
- **Input Channels:** 3 (RGB)
- **Output Classes:** 1 (Binary mask)
- **Loss Function:** `BCEWithLogitsLoss + DiceLoss`
- **Best Checkpoint:** Epoch 38
- **Best Validation IoU:** `0.6751` (`67.51%`)

---

## 📁 Project Structure

```
camouflaged-object-detection/
│
├── best_model.pth              # Trained PyTorch model checkpoint (Epoch 38)
├── model.py                    # Model definition, loader & device management
├── preprocessing.py            # ImageNet transforms, resize & postprocessing
├── inference.py                # Standalone Python inference script & CLI
├── app.py                      # Modern Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── test_images/                # Sample camouflaged test photographs
│   ├── chameleon.jpg
│   ├── leaf_insect.jpg
│   └── screech_owl.jpg
│
└── outputs/                    # Output directory for predicted masks & overlays
```

---

## 🚀 Installation & Setup

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### Interactive Streamlit Web Application
```bash
python -m streamlit run app.py
```
*Open [http://localhost:8501](http://localhost:8501) in your browser.*

### Command-Line Interface (CLI)
```bash
# Single image
python inference.py --image test_images/chameleon.jpg --output_dir outputs/

# Batch folder
python inference.py --input_dir test_images/ --output_dir outputs/ --threshold 0.5
```
