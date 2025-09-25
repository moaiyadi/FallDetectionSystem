# 🧑‍🦽 Fall Detection System for Humans

A deep learning–based **Fall Detection System** using the **YOLOv11 object detection algorithm** and **OpenCV**.  
The system can detect human falls in real time from video streams, images, or webcams, and can be extended with an **alerting module** for elderly care and safety monitoring.

---

## 🚀 Features
- **Training**: Fine-tune YOLOv11 on a custom fall-detection dataset.  
- **Prediction**: Detect falls from **images, videos, or live webcam feeds**.  
- **Benchmarking**: Evaluate model performance on GPU.  
- **Alert System**: Extendable to send notifications (e.g., via SMS/Email).  
- **Real-time performance** with high accuracy.  

---

## 📂 Project Structure
fall-detection-system/
│── Yolov11.py # Train YOLOv11 model on dataset
│── predict.py # Run inference on images, videos, or webcam
│── benchmark.py # Benchmark trained model performance
│── validate.py # Validate trained model performance
│── requirements.txt # Project dependencies
│── README.md # Documentation
│── LICENSE # License information

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/moaiyadi/fall-detection-system.git
cd fall-detection-system 
```

---

## 🧱 Requirements

The project requires the following key dependencies:
- Python 3.8+
- ultralytics (YOLOv11)
- OpenCV
- PyTorch & Torchvision
- NumPy, Matplotlib
- Twilio (for alert system – optional)

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Technologies

This project is built using the following technologies:

- Python
- YOLOv11 (Ultralytics)
- OpenCV for image/video processing
- PyTorch & Torchvision for deep learning

---

## 📂 Dataset Preparation
For this project, I created a **custom dataset** and formatted it using [Roboflow](https://roboflow.com/) so it could be used with YOLO.

You can download the dataset directly using the following code:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")   
project = rf.workspace("falldetection-bwfdz").project("yolo-dataset-98swd")
version = project.version(2)
dataset = version.download("yolov11")
```