from ultralytics import YOLO

# Load a model
model = YOLO("C:/Unihack2025/ObDetector/venv/runs/detect/trainMacheta2/weights/best.pt")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=150)  # train the model

#train/box_loss: 1.48821,
# train/cls_loss: 1.35898,
# train/dfl_loss: 1.63336,
# performance: 0.61274