from ultralytics import YOLO

# Load a model
model = YOLO("path/to/your/custom/trained/model.pt")  # load a pretrained model (trained on a custom dataset)

# Customize validation settings
validation_results = model.val(data="path/to/your/custom/dataset.yaml", imgsz=640, batch=16, conf=0.75, iou=0.6, device="cpu")