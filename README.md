# MicroSight-DETR

> **Micro-object Sight-enhanced Detection Transformer**  
> An enhanced RT-DETR for small object detection in aerial imagery

## 📁 Project Structure

```
MicroSight-DETR/
├── microsight_detr.yaml    # Model configuration file
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📄 File Descriptions

### `microsight_detr.yaml`
**Purpose**: Complete model architecture configuration for MicroSight-DETR

**Key Components**:
- **GEM Backbone**: Global Efficient Modeling blocks for feature extraction
- **MAFD Encoder**: Multi-domain Adaptive Fusion Dynamics for enhanced attention
- **SPAM Neck**: Spatial Preserving Aggregation with Multi-scale for small object optimization

**Usage**: Load this configuration file when training or inferencing with Ultralytics framework.

---

### `requirements.txt`
**Purpose**: Python package dependencies

**Main Dependencies**:
- PyTorch ≥ 1.8.0
- Torchvision ≥ 0.9.0
- Ultralytics (YOLO framework)
- OpenCV, NumPy, Matplotlib, etc.

**Installation**:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

### VisDrone2019 Object Detection Dataset

**Official Download Links**:
- **GitHub Repository**: [https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- **Official Website**: [http://aiskyeye.com/](http://aiskyeye.com/)
- **Direct Download** (Recommended):
  - Training Set: [VisDrone2019-DET-train](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing)
  - Validation Set: [VisDrone2019-DET-val](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing)  
  - Test-Dev Set: [VisDrone2019-DET-test-dev](https://drive.google.com/file/d/1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V/view?usp=sharing)

**Dataset Details**:
- **Classes**: 10 categories (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Training Images**: 6,471 images
- **Validation Images**: 548 images  
- **Test Images**: 3,190 images
- **Annotation Format**: YOLO format supported

**Dataset Structure**:
```
VisDrone2019-DET/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    └── images/
```

---

## 🏁 Pretrained Weights

📥 **Download Pretrained Weights**:  
The trained model weights (`best.pt`) are available on Google Drive:  
👉 [Download best.pt](https://drive.google.com/file/d/13SnZDrXrcgZ4G84qtC7CLXF3HHz37Hyv/view?usp=drive_link)

**Instructions**:
1. Download the `best.pt` file from the link above.
2. Place it in your project directory or specify the path during inference:
```bash
yolo predict model=best.pt source=path/to/images imgsz=640
```

---

## 🔬 Training Details & Reproducibility

### Training Time (Wall-Clock)

| Item | Value |
|---|---|
| Full 200-epoch training | ~18.5 hours |
| Average time per epoch | ~5.5 minutes |
| Hardware | Single NVIDIA RTX 5090 (32GB) |
| Batch size | 8 |
| Input resolution | 640×640 |

### Dataset Preprocessing Pipeline

**Training preprocessing:**

| Step | Operation | Details |
|---|---|---|
| 1 | Image resizing | Bilinear interpolation to 640×640 pixels (maintaining aspect ratio with padding) |
| 2 | Normalization | ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| 3a | Random horizontal flip | p=0.5 |
| 3b | Mosaic augmentation | p=0.5, following YOLOv5 implementation |
| 3c | Random HSV color jittering | hue=0.015, saturation=0.7, value=0.4 |
| 4 | Tensor conversion | HWC→CHW format, uint8→float32 |
| 5 | Bounding box normalization | Coordinates normalized to [0, 1] relative to image dimensions |

**Validation / Inference preprocessing:** Steps 1, 2, and 4 only (no augmentation).

### Statistical Evaluation Protocol

All experiments were conducted across **three independent runs** with different random seeds (**42, 1897, 7538**). Mean ± standard deviation are reported for all metrics. Full details are provided in Section 4.4.1 of the manuscript.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Download VisDrone2019 dataset from the links above and organize according to the structure shown.

### 3. Configure Dataset Path
Create a `VisDrone.yaml` file:
```yaml
path: /path/to/VisDrone2019-DET
train: train/images
val: val/images
test: test/images

nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```

### 4. Train Model
```bash
yolo train model=microsight_detr.yaml \
           data=VisDrone.yaml \
           epochs=200 \
           batch=8 \
           imgsz=640 \
           device=0
```

### 5. Inference
```bash
yolo predict model=path/to/best.pt \
             source=path/to/images \
             imgsz=640
```

---

## 💡 Features

- ✅ **Small Object Focused**: Optimized for detecting small objects in aerial imagery
- ✅ **High Performance**: Enhanced feature extraction and attention mechanisms
- ✅ **Easy to Use**: Based on Ultralytics framework with simple YAML configuration
- ✅ **Ablation Ready**: Modular design supports component-wise evaluation

---

## 📝 Citation

If you use MicroSight-DETR in your research, please cite:

```bibtex
@article{microsight2024,
  title={MicroSight-DETR: Enhanced Real-Time Detection Transformer for Small Object Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 📧 Contact

For questions and issues, please open an issue in this repository.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO framework
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) - Dataset provider
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) - Base architecture

---

## 📄 License

This project is released under the AGPL-3.0 License.
```
