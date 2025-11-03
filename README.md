```markdown
# MicroSight-DETR

A lightweight real-time detection transformer with triple enhancement for UAV-based micro-object detection.

---

## 📁 Repository Structure

```
MicroSight-DETR/
├── Template_for_submissions_to_Scientific_Reports.pdf
├── guo-FangAnYi-SOEP-backbone-Pola-SEFN-Mona-DyT.yaml
└── requirements.txt
```

---

## 📄 Files Description

### 📖 `Template_for_submissions_to_Scientific_Reports.pdf`
**Research Paper Manuscript**
- Complete academic paper with methodology and experimental results
- Describes the MicroSight-DETR architecture (GEM + MAFD + SPAM modules)
- Published version with performance analysis on VisDrone2019 dataset
- Contains ablation studies and comparative experiments

### ⚙️ `guo-FangAnYi-SOEP-backbone-Pola-SEFN-Mona-DyT.yaml`
**Model Configuration File**
- Defines the complete MicroSight-DETR architecture
- Specifies three core modules:
  - **GEM Block**: Global Efficient Modeling backbone
  - **MAFD Module**: Multi-domain Adaptive Fusion Dynamics encoder
  - **SPAM Module**: Spatial Preserving Aggregation with Multi-scale neck
- Used for model instantiation and training with Ultralytics framework
- Contains detailed comments on architecture design and parameter settings

### 📦 `requirements.txt`
**Python Dependencies**
- Lists all required Python packages with versions:
  - Deep learning frameworks (PyTorch, torchvision)
  - Computer vision libraries (OpenCV, Pillow)
  - Scientific computing tools (NumPy, SciPy, pandas)
  - Visualization dependencies (Matplotlib, Seaborn)
- Install via: `pip install -r requirements.txt`

---

## 📊 Dataset: VisDrone2019

### Official Download
**Primary Source**: [VisDrone-Dataset GitHub Repository](https://github.com/VisDrone/VisDrone-Dataset)

### Dataset Overview
- **Images**: 10,209 static images captured from drone platforms
  - Training: 6,471 images
  - Validation: 548 images
  - Testing: 3,190 images
- **Categories**: 10 object classes (pedestrian, person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Scenarios**: Diverse urban and suburban environments across 14 Chinese cities
- **Annotations**: Bounding boxes in PASCAL VOC and COCO formats

### Quick Setup with Ultralytics
The dataset is pre-configured in Ultralytics YOLO:
- Configuration file: [`VisDrone.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml)
- Documentation: [Ultralytics VisDrone Guide](https://docs.ultralytics.com/datasets/detect/visdrone/)

### Citation
If you use VisDrone2019 dataset, please cite:
```bibtex
@article{zhu2021detection,
  title={Detection and tracking meet drones challenge},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={11},
  pages={7380--7399},
  year={2021},
  publisher={IEEE}
}
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
Follow instructions at [VisDrone GitHub](https://github.com/VisDrone/VisDrone-Dataset) to download the dataset.

### 3. Training
```bash
yolo train model=guo-FangAnYi-SOEP-backbone-Pola-SEFN-Mona-DyT.yaml \
           data=VisDrone.yaml \
           epochs=200 \
           batch=8 \
           imgsz=640 \
           device=0
```

---

## 📈 Performance Highlights

| Metric | Baseline RT-DETR | MicroSight-DETR | Improvement |
|--------|------------------|-----------------|-------------|
| mAP@0.5 | 46.7% | **51.3%** | +4.6% |
| mAP@0.5:0.95 | 28.4% | **31.8%** | +3.4% |
| Precision | 60.3% | **63.3%** | +3.0% |
| Recall | 45.2% | **49.7%** | +4.5% |
| FPS | 116 | 78 | Real-time capable |

---

## 📝 Citation

If you use this work in your research, please cite:
```bibtex
@article{guo2025microsight,
  title={MicroSight-DETR: A Lightweight Real-time Detection Transformer with Triple Enhancement for UAV-based Logistics Monitoring},
  author={Guo, Junhao and Jiang, Jinhao and Yang, Zijing and Zhang, Hao},
  journal={Scientific Reports},
  year={2025}
}
```

---

## 📧 Contact

For questions or collaborations, please contact:
- Zijing Yang: yangzijing@bigc.edu.cn
- Hao Zhang: howzh@bigc.edu.cn

---

## 🙏 Acknowledgments

- Dataset: [VisDrone Team, Tianjin University](https://github.com/VisDrone/VisDrone-Dataset)
- Framework: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Funding: Beijing Institute of Graphic Communication Key Teaching Reform Project (No. 22150122009)

---

## 📄 License

This project follows the license terms specified in the original paper and dataset.
