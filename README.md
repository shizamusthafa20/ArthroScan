# ArthroScan: AI-Powered Pest Identification Dashboard 🐞🌾

ArthroScan is a premium, end-to-end pest identification system designed to help farmers and agronomists identify insect pests with high accuracy, understand their agricultural impact, and receive tailored treatment recommendations.

![ArthroScan Dashboard Preview](https://img.shields.io/badge/AI-ResNet18-green)
![Deployment](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-blue)

## ✨ Features

- **High-Accuracy Identification**: Custom-trained ResNet18 model achieving ~70% accuracy on a 30-class subset of the IP102 dataset.
- **AI Explainability**: Integrated **Grad-CAM** heatmaps that visualize exactly which parts of the image the AI focused on to make its prediction.
- **Agricultural Intelligence**: Comprehensive database of 30 pests including:
  - Scientific names and fun facts.
  - Affected crops and economic loss assessments.
  - **Tailored Recommendations**: Organic, Chemical, and Biological treatment options.
- **Human Hazard Assessment**: Instant safety evaluation (Venomous, Bites, Disease Carrier, Allergy risks).
- **Premium Glassmorphism UI**: A modern, dark-mode dashboard built with Vanilla HTML, CSS, and JS.
- **Unified Deployment**: FastAPI backend that serves the entire frontend and API from a single Docker container.

## 🚀 Tech Stack

- **Frontend**: HTML5, Vanilla CSS3 (Custom Design), JavaScript (ES6+)
- **Backend**: FastAPI (Python)
- **Deep Learning**: PyTorch, Torchvision
- **Explainability**: Grad-CAM (pytorch-grad-cam)
- **Deployment**: Docker, Hugging Face Spaces

## 🚀 Live Demo

You can try out the ArthroScan dashboard live on Hugging Face Spaces:
**[ArthroScan Live Dashboard](https://huggingface.co/spaces/shizamusthafa/ArthroScan)**

## 🛠️ Local Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shizamusthafa20/ArthroScan.git
   cd ArthroScan
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```


## 🧪 Training

The model was trained on the **IP102 Dataset**.
- **Local Training**: `train_local.py` (Fast CPU-based demo).
- **High-Accuracy Training**: `train_full_colab.py` (Designed for Kaggle/Colab GPUs).

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
