import os
import torch
import cv2
import time
import requests
from ultralytics import YOLO
from torchvision.ops import nms

API_BASE_URL = "http://56.228.19.103:8000"
PARKING_NUMBER = "100000053"

runs_folder = "C:/Unihack2025/ObDetector/venv/runs/detect"
fallback_video_path = "C:/Unihack2025/ObDetector/venv/test_videos/videoccc.mp4"

CAMERA_INDEX = 1

train_folders = [f for f in os.listdir(runs_folder) if f.startswith("train")]
train_folders.sort()
if not train_folders:
    raise ValueError(f"Niciun folder 'train' găsit în {runs_folder}")

last_train = train_folders[-1]
model_path = os.path.join(runs_folder, last_train, "weights", "best.pt")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Modelul {model_path} nu există!")

print(f"Folosește modelul: {model_path}")

model = YOLO(model_path)
print("Clase disponibile în model:")
print(model.names)

def simplify_class_name(name: str) -> str:
    name = name.lower()
    if "empty" in name:
        return "empty"
    elif "occupied" in name:
        return "occupied"
    else:
        return name

def send_to_api(free_count, occupied_count):
    total_spots = free_count + occupied_count
    payload = {
        "parking_number": int(PARKING_NUMBER),
        "free_spots": free_count,
        "total_spots": total_spots
    }
    try:
        response = requests.post(f"{API_BASE_URL}/api/detection", json=payload, timeout=5)
        if response.status_code == 200:
            print(f"Date trimise către API: {payload}")
        else:
            print(f"Eroare API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Eroare conexiune cu API: {e}")

def open_camera_source():
    print(f"Încerc să deschid camera la index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    time.sleep(0.2)
    if cap.isOpened():
        print("Camera conectată.")
        return cap
    else:
        print(f"Nu s-a putut deschide camera, folosesc fallback video.")
        cap = cv2.VideoCapture(fallback_video_path)
        return cap

cap = open_camera_source()
if not cap.isOpened():
    raise ValueError("Nu s-a putut deschide nicio sursă video.")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

update_interval = 1

frame_count = 0
last_update_time = 0
free_count = occupied_count = total_count = 0
last_detections = []

cv2.namedWindow("YOLO Parking Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Parking Detection", 1280, 720)

print("Pornim analiza în timp real...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Eroare la citirea frame-ului, retry...")
        time.sleep(0.5)
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Nu am putut citi frame după retry.")
            break

    frame_count += 1
    current_time = time.time()

    if current_time - last_update_time >= update_interval or not last_detections:
        last_update_time = current_time
        print("Actualizare YOLO...")
        results = model.predict(source=frame, conf=0.3, verbose=False)

        free_count = occupied_count = total_count = 0
        last_detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu()
            scores = r.boxes.conf.cpu()
            classes = r.boxes.cls.cpu()

            keep = nms(boxes, scores, iou_threshold=0.4)
            boxes = boxes[keep]
            scores = scores[keep]
            classes = classes[keep]

            for box, score, cls in zip(boxes, scores, classes):
                class_name = model.names[int(cls)]
                simple_name = simplify_class_name(class_name)
                x1, y1, x2, y2 = map(int, box)

                label = f"{simple_name} {float(score):.2f}"
                last_detections.append({
                    "box": (x1, y1, x2, y2),
                    "score": float(score),
                    "label": label,
                    "simple_name": simple_name
                })

                if simple_name in ["empty", "occupied"]:
                    total_count += 1
                    if simple_name == "empty":
                        free_count += 1
                    elif simple_name == "occupied":
                        occupied_count += 1

        print(f"Free={free_count}, Occupied={occupied_count}, Total={total_count}")
        send_to_api(free_count, occupied_count)

        del results
        torch.cuda.empty_cache()

    display_frame = frame.copy()
    cv2.rectangle(display_frame, (10, 10), (420, 80), (255, 255, 255), -1)
    text = f"Free: {free_count} | Occupied: {occupied_count} | Total: {free_count + occupied_count}"
    cv2.putText(display_frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    for det in last_detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        simple_name = det["simple_name"]
        color = (0, 255, 0) if simple_name == "empty" else (0, 0, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO Parking Detection", display_frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key in [ord('q'), 27]:
        print("Oprire manuală...")
        break

cap.release()
cv2.destroyAllWindows()
print("Analiza finalizată.")
