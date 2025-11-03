
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
           epochs=100 \
           batch=16 \
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
