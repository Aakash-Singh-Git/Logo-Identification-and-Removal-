# Logo Identification and Removal

A desktop application for automated identification and removal of logos (e.g. Zoom watermark for Times Network) from images using a YOLO-based detector and multiple AI-powered inpainting backends, with an OpenCV fallback.

---

## 🚀 Features

- **YOLOv8-based Detection**  
  - Real-time logo detection (Nano/Small variant)  
  - Data augmentation pipeline for robust training  

- **Multi-Backend Inpainting**  
  - Stability AI, OpenAI DALL·E 2, Replicate (Stable Diffusion), Hugging Face, ClipDrop  
  - OpenCV Telea algorithm fallback for offline use  

- **User-Friendly GUI**  
  - Built with Python Tkinter  
  - Live preview of detection masks and inpainting results  
  - Configurable AI method selection  

- **Extensible Architecture**  
  - Modular codebase (see `structure.py`, `yolo_model.py`, `data_create.py`, `Logo Remover Main.py`)  
  - Easily retrain on new logos or integrate additional services  

---

.
- ├── data_create.py            # Synthetic dataset generation & annotation
- ├── structure.py              # Dataset organization & train/val split
- ├── yolo_model.py             # YOLOv8 model loading & inference wrappers
- ├── Logo Remover Main.py      # Tkinter GUI & orchestrator for detection + inpainting
- ├── requirements.txt          # Python dependencies (including ultralytics==8.0.114)
- ├── data/                     # Place your images and labels here
- ├── config.json               # API keys & paths for inpainting services
- └── api_keys.json             # (gitignored) Service credentials


## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/logo-remover.git
   cd logo-remover
