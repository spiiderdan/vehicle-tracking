# Vehicle Detection & Tracking Project

This repository provides a complete **end-to-end pipeline** for real-time vehicle detection, refinement, and tracking using a hybrid YOLOv8 + Faster R-CNN detector integrated with DeepSORT. It’s organized as a professional project you can run, explore, or adapt for your own datasets.

---

## 📂 Directory Structure

```
vehicle_detection_project/
│
├── data/
│   ├── voc_format/            
│   │   ├── train/             # Pascal VOC images + XML annotations for Faster R-CNN  
│   │   └── valid/             
│   └── yolo_format/           # YOLOv8–style images + .txt labels  
│       ├── train/             
│       ├── valid/             
│       └── test/              
│
├── models/
│   ├── best.pt                # Trained YOLOv8 weights  
│   ├── faster_rcnn_final.pth  # Final Faster R-CNN weights  
│   └── checkpoints/           # RCNN epoch‐by‐epoch checkpoints  
│
├── scripts/
│   ├── train_yolov8.py        # YOLOv8 fine-tuning script  
│   └── train_rcnn.py          # Faster R-CNN fine-tuning script  
│
├── notebooks/
│   └── final_project.ipynb    # Jupyter notebook with full pipeline  
│
├── outputs/
│   ├── tracked_output.mp4     # Video with hybrid detection & tracking overlay  
│   └── tracking_report.txt    # Summary report (total counts & lifespans)  
│
└── README.md                  # Project overview & instructions  
```

---

## ⚙️ Prerequisites

- **Python 3.8+**  
- **PyTorch 1.12+** & **torchvision 0.13+**  
- **ultralytics** (YOLOv8)  
- **opencv-python**  
- **deep_sort_realtime**  
- **tqdm**, **Pillow**, **numpy**

Install dependencies:

```bash
pip install torch torchvision ultralytics opencv-python deep_sort_realtime tqdm Pillow numpy
```

---

## 🚀 Usage

1. **Prepare your data**  
   - VOC-style images + XML under `data/voc_format/{train,valid}`  
   - YOLO-style images + TXT under `data/yolo_format/{train,valid,test}`  

2. **Train or skip**  
   - **YOLOv8:**  
     `python scripts/train_yolov8.py`  
     Final weights → `models/best.pt`  
   - **Faster R-CNN:**  
     `python scripts/train_rcnn.py`  
     Final weights → `models/faster_rcnn_final.pth` (checkpoints in `models/checkpoints/`)  

3. **Explore in Notebook**  
   Open `notebooks/final_project.ipynb` to follow step by step:
   - Model loading  
   - Conditional training  
   - Hybrid detection & refinement  
   - Tracking & logging  
   - Visualization & metrics  

4. **Run full pipeline**  
   `python scripts/run_hybrid_tracking.py`  
   Outputs:  
   - Overlay video → `outputs/tracked_output.mp4`  
   - Text report → `outputs/tracking_report.txt`

---

## 🎥 Output Video

`outputs/tracked_output.mp4`

---

## 📑 Tracking Report

A human-readable summary saved in `outputs/tracking_report.txt`, for example:

```
=== Tracking Report ===

Video processed: inputs/traffic_video.mp4
Total frames: 1200

BUS:
  • Total unique seen : 23
  • Avg lifespan      : 45.2 frames

CAR:
  • Total unique seen : 58
  • Avg lifespan      : 60.8 frames

TRUCK:
  • Total unique seen : 15
  • Avg lifespan      : 50.1 frames

VAN:
  • Total unique seen : 10
  • Avg lifespan      : 40.5 frames
```

---

## 🔧 Customization

- **Thresholds:** Adjust `yolo_confirm_thresh`, `rcnn_conf_thresh`, NMS IoUs in the notebook  
- **Tracker settings:** Modify `max_age`, `n_init` in the tracking section  
- **Data paths:** Update variables in scripts/notebook  

---

## 📄 License & Acknowledgments

This project is for academic and demonstration purposes. Adapt & extend as you wish—please cite any reused code or datasets. Enjoy your vehicle detection & tracking pipeline!